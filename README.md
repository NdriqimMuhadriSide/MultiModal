# MultiModal

A multimodal AI assistant: chat, document Q&A, image and audio understanding, and live screen/camera streaming — behind one FastAPI backend and one offline-capable Next.js client.

Everything runs locally and on free tiers. Embeddings are computed on-device, the vector store and both databases are local files, and the LLM is reached through Groq's free API.

---

## Architecture

```
   User (web / mobile browser)
   text · documents · audio · image · video
                 |
                 v
   ┌─────────────────────────────┐        ┌──────────────────────────┐
   │  Frontend — Next.js         │ <────> │  Client-side storage     │
   │  · config / API client      │        │  5 tiers, see below      │
   │  · services · Zustand store │        │  + service worker        │
   │  · UI components            │        └──────────────────────────┘
   └─────────────────────────────┘
                 |  HTTP  /api/v1/*
                 v
   ┌─────────────────────────────────────────────────────────┐
   │  Backend — FastAPI                                       │
   │                                                          │
   │  processors/  frame sampling, stream sessions            │
   │  rag/         loaders → layout → structure → splitter →   │
   │               embeddings → ChromaDB ─┐                     │
   │              (× N query phrasings)   │                     │
   │                            BM25 ─────┴→ fusion → rerank → │
   │                                        grade → answer/refuse│
   │               context builder                             │
   │  agents/      LangGraph StateGraph (route → tool → retry) │
   │  memory/      conversation history (SQLite)              │
   │  ai/          LLM · vision · transcription services      │
   └─────────────────────────────────────────────────────────┘
                 |
                 v
        Groq / OpenAI-compatible LLM API
```

## Stack

**Backend** — Python 3.12, FastAPI, Pydantic Settings, LangGraph, ChromaDB, sentence-transformers (local embeddings), SQLite, pytest.

**Frontend** — Next.js (App Router), React, TypeScript, Tailwind CSS, Zustand, a hand-written service worker.

> Python 3.13+ is not supported: a transitive `tokenizers` dependency has no prebuilt wheel and fails to build from source.

---

## Supported document formats

| Format | How structure is found |
| --- | --- |
| `.pdf` | Inferred from glyph positions — reading order, columns, tables, running headers (`rag/layout.py`); scanned pages fall back to OCR (`rag/ocr.py`) |
| `.docx` | Read from paragraph styles ("Heading 2") and real tables |
| `.html` / `.htm` | Read from `<h1>`–`<h6>` and `<table>`; scripts, styles and `<head>` dropped |
| `.md` / `.markdown` / `.txt` | Read from `#` and underlined headings; code fences passed through untouched |
| `.csv` / `.tsv` | The whole file is a table; rows are emitted in groups that repeat the header so every chunk keeps its column names |

Every loader produces the same thing — blocks in reading order, with headings marked and each block tagged with the section it sits under — so the rest of the pipeline is format-agnostic. Adding a format means one module in `rag/loaders/`.

## Chunking strategies

`CHUNKING_STRATEGY` selects how a document is cut and what each piece is used for. All of them produce the same records, so ingestion, storage and retrieval are unchanged whichever is picked.

| Strategy | Embeds | Returns | Cost |
| --- | --- | --- | --- |
| `recursive` *(default)* | the chunk | the chunk | free |
| `semantic` | the chunk | the chunk | one embedding per sentence |
| `sentence_window` | one sentence | the sentence plus neighbours | large index |
| `parent_document` | a small child | the whole parent | large index |
| `propositional` | an LLM-written atomic fact | the source passage | one LLM call per passage |

`CONTEXTUAL_RETRIEVAL=true` layers Anthropic's technique on top of any of them: an LLM writes one line situating each chunk in its document, prepended before embedding. Off by default — it costs an LLM call per chunk.

Tables never split without repeating their header row, and chunks never straddle a heading, whichever strategy is in use.

Each document's own metadata (title, author, date) is read separately by `rag/metadata.py` — from the PDF `/Info` dictionary, `docProps/core.xml`, HTML `<meta>` tags, or Markdown front matter — and copied onto every chunk, so `/documents/search` can filter on it. Similarity alone can't answer "anything by the compliance team": no embedding encodes an author.

## Retrieval

`RETRIEVAL_MODE` selects how a question finds chunks. The default is `hybrid` — both halves run and their rankings are fused.

| Mode | How it finds chunks | Good at | Blind to |
| --- | --- | --- | --- |
| `dense` | Embed the question, cosine search in Chroma | Paraphrase — "how do I stop paying" finds the cancellation clause | Rare literal tokens: `ERR-4021`, a part number, a surname |
| `keyword` | BM25 over chunk text (`rag/keyword_index.py`) | Exact terms, identifiers, acronyms, numbers | Anything worded differently from the document |
| `hybrid` *(default)* | Both, merged by reciprocal rank fusion (`rag/fusion.py`) | Both of the above | — |

The two halves fail in opposite directions, and which kind a question is isn't knowable in advance — so the robust move is to ask both. Fusion uses rank rather than score, because a cosine similarity and a BM25 score have no common scale and normalising each list to `[0, 1]` would make the best result of a list of garbage look like a perfect match. A chunk both halves surfaced outranks a chunk either one ranked first, which is exactly the signal worth having.

The keyword half costs no API key and no model: it is an inverted index built in memory from the chunks already in Chroma, rebuilt on the first query after any ingest or delete. It reads the corpus back out of the vector store rather than keeping a copy, so the two halves can never disagree about what has been ingested.

### Reranking

`RERANK_ENABLED=true` (default) adds a second scoring pass over the shortlist retrieval produced. A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB, on-device, no API key) reads the question and each candidate *together* and scores how well that chunk answers it.

The embedding model can't do this: chunk vectors are computed at ingestion, before any question exists, so it can only judge "are these two texts about similar things" — never "does this answer that". The cross-encoder can, but nothing about it can be precomputed, so it's only viable over ~20 candidates. Hence the funnel: retrieval over-fetches for recall, reranking narrows for precision.

Measured on a 39-chunk, 24-question set (`Hit@1 / Recall@3 / MRR`):

| | Hit@1 | Recall@3 | MRR | p50 latency |
| --- | --- | --- | --- | --- |
| dense only | 91.7% | 95.8% | 0.946 | 5.8 ms |
| hybrid | 79.2% | 95.8% | 0.875 | 4.2 ms |
| hybrid + rerank | **91.7%** | **100%** | **0.951** | 19.2 ms |

### Multi-query retrieval

`MULTI_QUERY_ENABLED=true` (default **off**) asks an LLM for `MULTI_QUERY_COUNT` alternative phrasings of the question, retrieves for each, and fuses every ranking together. A chunk several phrasings agree on rises; a chunk one odd rewrite found alone gets a single vote. The original question is always retrieved for, and always first — so a rewrite that paraphrases `ERR-4021` into "upload errors" can't take the right chunk away, it can only fail to add one.

It is off by default because measurement said so, not caution. On the same 24-question set:

| | Hit@1 | Recall@3 | Recall@5 | MRR | retrieval p50 |
| --- | --- | --- | --- | --- | --- |
| hybrid | 83.3% | 95.8% | 95.8% | 0.896 | 3.8 ms |
| hybrid + multi-query | 79.2% | 100% | 100% | 0.896 | 18.4 ms |
| hybrid + rerank | 91.7% | 100% | 100% | 0.951 | 19.1 ms |
| hybrid + rerank + multi-query | 91.7% | 100% | 100% | 0.951 | 34.1 ms |

Multi-query does fix recall on its own (Recall@5 95.8% → 100%), but reranking had already closed that gap, so the two together are identical to reranking alone — for an extra ~350ms LLM call per question. The `found_by_query` field on each source says why: across 24 questions, the correct chunk was surfaced by a variant rather than the original **0 times**. Turn it on for a corpus large enough that Recall@5 is genuinely short of 100%, and check `found_by_query` to confirm it is earning its call.

Reranking also makes `HYBRID_DENSE_WEIGHT` / `HYBRID_KEYWORD_WEIGHT` stop mattering — across weights from 0 to 2 the reranked results are identical, because fusion only has to get the right chunk *into* the shortlist and the cross-encoder redoes the order. Those weights are worth tuning only with `RERANK_ENABLED=false`.

### Corrective RAG

`CORRECTIVE_RAG_ENABLED=true` (default **off**) grades what retrieval produced before answering from it, and acts on the grade — answer, retry with rephrased queries, or **refuse**. It is the only thing in the pipeline that can decline to answer: everything else makes retrieval better at finding something, this notices when what it found isn't worth answering from.

The grade is the cross-encoder's score on the best candidate, so it needs `RERANK_ENABLED`. An LLM is optional — it powers the retry only; grading and refusal need none.

Measured over 24 answerable + 10 deliberately unanswerable questions (the guest network exists but its password doesn't; `ERR-4021` exists but `ERR-5000` doesn't):

| | Hit@1 | false refusals | correct refusals | overall |
| --- | --- | --- | --- | --- |
| hybrid + rerank | 91.7% | 0% | 10% | 67.6% |
| + corrective (`reject=0.001`) | 83.3% | 12.5% | **70%** | **79.4%** |
| + corrective (`reject=0.02`) | 75.0% | 25.0% | 80% | 76.5% |
| + corrective (`reject=0.4`) | 58.3% | 41.7% | 90% | 67.6% |

It is a genuine trade, not a free win, which is why it is off by default. Refusals go from 10% to 70% and 12.5% of answerable questions get wrongly refused. The break-even is roughly **12% unanswerable traffic** — above that it wins, below it costs more answers than it saves wrong ones.

Two things the measurement settled. The cross-encoder is an excellent *ranker* and a poorly calibrated *classifier*: some answerable questions score `0.0001` even with the correct chunk ranked first, while "how do I fix ERR-5000?" scores `0.65`, so no threshold separates them cleanly. And the rephrase-and-retry step changed **nothing** — grading alone produced identical numbers — which is why the LLM is optional rather than required.

### Agentic RAG

Two changes, both about the *conversation* rather than the index.

**Retrieval now sees the conversation.** It never did: the embedding model gets one string and BM25 matches the terms in one string, so a follow-up like "how far ahead do I have to request it?" was searched for as six words with no idea what "it" meant. `rag/contextualizer.py` rewrites it into a standalone question first — "how far ahead do I have to request annual leave?" — and retrieves for that. The user's own wording still reaches the answering prompt; only the search term changes. The call is skipped entirely when there is no history, so `/rag/ask` never pays for it.

**The agent is no longer single-hop.** When the knowledge-base tool retrieves nothing, a conditional edge routes to a second node that answers from the model's own knowledge and says so (`tool_used: answer_directly_after_knowledge_base`). Before, a mis-route to `KNOWLEDGE_BASE` was terminal — the router's one guess was also its last.

Measured on 10 conversation pairs, scoring where the follow-up's correct chunk ranks:

| | Hit@1 | Recall@3 | MRR |
| --- | --- | --- | --- |
| retrieval without history | 90.0% | 90.0% | 0.900 |
| with contextualization | 90.0% | **100%** | **0.950** |
| *single-turn no-damage check* | 91.7% → 91.7% | 100% → 100% | 0.951 → 0.951 |

A modest, real gain — and a cautionary one. The first version of the rewrite prompt scored **30% Hit@1**, far worse than doing nothing, because the model absorbed the previous *answer* into the question: "and VPN-7731?" became "...in relation to the 25 MB size cap that causes ERR-4021?", which retrieved the ERR-4021 chunk. Naive contextualization is actively harmful; the prompt has to insist on changing as few words as possible and adding no facts from earlier turns.

`POST /documents/search` stays pure dense on purpose — it is the raw similarity probe, and its metadata filters (author, title, date) are a property of the vector store's query, not of the text index.

Retrieval is exposed on `/rag/ask`: each source reports `similarity` (cosine, `null` when only keyword search found it), `keyword_score`, and `matched_by` — so "the model cited a chunk dense search never saw" is visible rather than inferred.

## Getting started

### Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then add your GROQ_API_KEY
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Get a free key at [console.groq.com](https://console.groq.com) → API Keys. `OPENAI_API_KEY` is optional — Groq is the active provider for chat and vision.

**Optional — OCR for scanned PDFs.** `pytesseract` wraps the Tesseract engine but doesn't bundle it, so the binary is a separate install:

```bash
brew install tesseract                  # macOS
sudo apt-get install tesseract-ocr      # Debian/Ubuntu
```

Skip it and everything still runs; PDFs whose pages are images are recorded as `FAILED` with a reason saying OCR isn't installed.

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

http://localhost:3000

### Tests

```bash
cd backend && .venv/bin/pytest       # backend
cd frontend && npx tsc --noEmit && npx eslint .   # frontend
```

---

## API

All routes are under `/api/v1`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness — deliberately never cached |
| `POST` | `/chat` | Chat with conversation memory |
| `POST` | `/chat/stream` | Same, streamed token by token over SSE |
| `GET` | `/chat/{conversation_id}/history` | Full stored history for a conversation |
| `POST` | `/agent/ask` | LangGraph agent — routes to a tool, then answers |
| `POST` | `/agent/ask/stream` | Same, streamed over SSE (this is what the UI uses) |
| `POST` | `/documents/upload` | Upload a document (PDF, DOCX, HTML, Markdown, CSV, TXT) |
| `POST` | `/documents/ingest` | Chunk, embed, and index a document |
| `GET` | `/documents` | List ingested documents |
| `POST` | `/documents/search` | Raw similarity search over chunks, filterable by author/title/date (dense only) |
| `POST` | `/rag/ask` | Retrieval-augmented answer, with per-source retrieval scores |
| `POST` | `/rag/chat` | RAG with conversation memory |
| `POST` | `/vision/analyze` | Describe or answer questions about an image |
| `POST` | `/audio/analyze` | Transcribe and reason over audio |
| `POST` | `/stream/frame` | Submit one sampled frame from a live stream |
| `POST` | `/stream/{session_id}/end` | Tear down a stream session |

---

## Memory

Two independent layers, deliberately not synchronized:

**Short-term (backend)** — `memory/conversation_memory.py`. SQLite, keyed by `conversation_id`. Every turn is stored permanently; only the most recent `CONVERSATION_HISTORY_LIMIT` (default 10) are replayed into the prompt. The cap is a view limit, not a retention limit.

**Session context (backend)** — an in-process ring buffer of recent stream observations per session, bounded and TTL-expired. Volatile by design.

There is no long-term memory: nothing is keyed by user, and nothing is summarized or promoted across conversations.

## Client-side storage

Five tiers in the browser, each chosen for a property the others lack:

| Tier | Holds | Lifetime |
| --- | --- | --- |
| `localStorage` | chat list (cap 50), theme, onboarding | Permanent |
| `sessionStorage` | unsent drafts, scroll position, active chat | Per tab |
| Cookie | appearance hint for SSR | Permanent, sent with requests |
| Cache API | uploaded image/audio bytes, offline outbox | Until evicted |
| IndexedDB | full message archive, searchable, uncapped | Permanent |

Chat history lives **only** in the browser — the client never fetches it from the backend.

A service worker (`frontend/public/sw.js`, production only) serves hashed assets cache-first, pages network-first with a cached fallback, and `/api/v1/documents` stale-while-revalidate. All POSTs always hit the network; when offline they queue in an outbox and replay via Background Sync on reconnect.

---

## Layout

```
backend/
  app/          FastAPI app, config, routes, services
  agents/       LangGraph assistant agent
  ai/           LLM, vision, transcription services
  rag/          loaders → blocks → chunks → embeddings → Chroma → retrieval
    loaders/    one module per format (pdf, docx, html, markdown, csv)
  memory/       conversation memory (SQLite)
  processors/   streaming frame sampling and session state
  prompts/      system and routing prompts
  tests/        pytest suite
frontend/
  app/          App Router routes
  components/   UI
  hooks/        feature hooks (chat, vision, audio, streaming, …)
  lib/          API client, storage tiers, validators
  services/     one module per backend feature
  store/        Zustand stores
  public/sw.js  service worker
```

## Configuration

Every setting is an environment variable with a sane default — see `backend/.env.example` for the full annotated list. The ones that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required |
| `CONVERSATION_HISTORY_LIMIT` | `10` | Turns replayed into each prompt |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `256` / `48` | Chunk size in embedding-model **tokens**, clamped to the model's limit |
| `PDF_LAYOUT_MODE` | `true` | Layout-aware PDF parsing (columns, tables, headers) |
| `OCR_ENABLED` / `OCR_LANGUAGE` | `true` / `eng` | OCR fallback for scanned PDFs — needs the Tesseract binary (optional, see below) |
| `CHUNK_SECTION_HEADERS` | `true` | Prefix each chunk with the heading path it came from |
| `CHUNKING_STRATEGY` | `recursive` | `recursive`, `semantic`, `sentence_window`, `parent_document`, `propositional` |
| `CONTEXTUAL_RETRIEVAL` | `false` | LLM-written context line per chunk — costs one call per chunk |
| `RETRIEVAL_MODE` | `hybrid` | `dense`, `keyword`, `hybrid` (dense + BM25, fused by rank) |
| `HYBRID_DENSE_WEIGHT` / `HYBRID_KEYWORD_WEIGHT` | `1.0` / `1.0` | How much each half counts for in the fused ranking; `0` disables a half |
| `RERANK_ENABLED` | `true` | Cross-encoder second pass over the shortlist — local, ~80MB, ~15ms/query |
| `MULTI_QUERY_ENABLED` | `false` | LLM-generated question phrasings, all retrieved and fused — costs one LLM call per question |
| `CORRECTIVE_RAG_ENABLED` | `false` | Grade retrieval and refuse when nothing relevant was found — needs `RERANK_ENABLED` |
| `QUERY_CONTEXTUALIZATION_ENABLED` | `true` | Rewrite follow-ups into standalone questions before retrieving |
| `AGENT_KB_FALLBACK_ENABLED` | `true` | Agent answers from general knowledge (disclosed) when the documents have nothing |
| `MAX_UPLOAD_SIZE_MB` | `25` | PDF upload ceiling |
| `STREAM_SAMPLING_INTERVAL_SECONDS` | `2.0` | One frame every N seconds |
| `CORS_ORIGINS` | `localhost:3000, localhost:8080` | Comma-separated |

Local data (`backend/data/`) — SQLite databases, the Chroma index, and uploads — is gitignored and regenerated on first run.
