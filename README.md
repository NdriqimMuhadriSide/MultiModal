# MultiModal

A multimodal AI assistant: chat, document Q&A, image and audio understanding, and live screen/camera streaming — behind one FastAPI backend and one offline-capable Next.js client.

A supervisor agent sits in front of all of it. Rather than the caller picking between `/chat`, `/rag/chat` and `/vision/ask`, one message goes to `/agent/ask` and the supervisor decides whether it can answer directly or needs a specialist — and, when a question spans both a picture and the corpus, uses more than one.

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
   ┌───────────────────────────────────────────────────────────┐
   │  Backend — FastAPI                                        │
   │                                                           │
   │  agents/      supervisor ─┬→ research agent ─┐            │
   │               (ReAct loop)├→ vision agent ───┤            │
   │                           └→ external API    │            │
   │               critic gates the answer; one shared budget  │
   │  rag/         loaders → layout → structure → splitter →   │
   │               embeddings → ChromaDB ─┐                    │
   │              (× N query phrasings)   │                    │
   │                            BM25 ─────┴→ fusion → rerank → │
   │                                        grade → answer     │
   │  memory/      history · compaction · attachments (SQLite) │
   │  processors/  frame sampling, stream sessions             │
   │  ai/          LLM · vision · transcription services       │
   │  a2a/         Agent Card + JSON-RPC, so other agents can  │
   │               call the research agent over the network    │
   └───────────────────────────────────────────────────────────┘
                 |
                 v
        Groq / OpenAI-compatible LLM API
```

## Stack

**Backend** — Python 3.12, FastAPI, Pydantic Settings, ChromaDB, sentence-transformers (local embeddings), SQLite, pytest. No agent framework: the reasoning loop is ~800 hand-written lines (`agents/agent_loop.py`) — see [Agents](#agents) for why.

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

**The agent is no longer single-hop.** A mis-route used to be terminal — the router's one guess was also its last. It now loops, and searches again on the strength of what the last search returned; see [Agents](#agents).

Measured on 10 conversation pairs, scoring where the follow-up's correct chunk ranks:

| | Hit@1 | Recall@3 | MRR |
| --- | --- | --- | --- |
| retrieval without history | 90.0% | 90.0% | 0.900 |
| with contextualization | 90.0% | **100%** | **0.950** |
| *single-turn no-damage check* | 91.7% → 91.7% | 100% → 100% | 0.951 → 0.951 |

A modest, real gain — and a cautionary one. The first version of the rewrite prompt scored **30% Hit@1**, far worse than doing nothing, because the model absorbed the previous *answer* into the question: "and VPN-7731?" became "...in relation to the 25 MB size cap that causes ERR-4021?", which retrieved the ERR-4021 chunk. Naive contextualization is actively harmful; the prompt has to insist on changing as few words as possible and adding no facts from earlier turns.

`POST /documents/search` stays pure dense on purpose — it is the raw similarity probe, and its metadata filters (author, title, date) are a property of the vector store's query, not of the text index.

Retrieval is exposed on `/rag/ask`: each source reports `similarity` (cosine, `null` when only keyword search found it), `keyword_score`, and `matched_by` — so "the model cited a chunk dense search never saw" is visible rather than inferred.

---

## Agents

`POST /agent/ask` used to be a router: one LLM call picked one of three tools, that tool ran, and the answer was whatever came back. That shape can't answer

> "Does this receipt comply with our expense policy?"

which needs the image read **and** the corpus searched, in that order, with the second question shaped by what the first returned. A router that picks exactly one tool never could, and no single specialist can either. So the router became a supervisor.

### The loop

Every agent shares one hand-written ReAct loop (`agents/agent_loop.py`):

```
build prompt (question + tools + scratchpad)
  → ask the LLM for one Thought + one Action
  → parse the Action out of free text
  → run the tool → Observation
  → append (Thought, Action, Observation) ─┐
  ←────────────────────────────────────────┘  until a terminal tool, or the budget runs out
```

There is no hidden state. The LLM is stateless between calls, so "what have I already tried" exists only because the loop keeps the scratchpad and re-sends it whole every turn.

`langgraph` was a dependency while the old router existed and was **dropped**, not replaced. What the loop actually needed turned out to be a brace-scanning parser that absorbs every way a model deviates from the format it was given, three independent stop conditions (step ceiling, consecutive-parse-failure ceiling, forced synthesis), and turning a tool's exception into an observation so a failing tool costs a step rather than the run. None of that is what a graph library provides, and a framework in between makes each of them harder to see.

### Who does what

| Agent | Tools | For |
| --- | --- | --- |
| **Supervisor** (`supervisor_agent.py`) | `research_documents`, `read_image`, `call_external_api`, `finish` | Deciding. Answers directly when it can; delegates when it can't; calls more than one specialist when a question spans both |
| **Research** (`research_agent.py`) | `list_documents`, `search_knowledge_base`, `finish` | Multi-hop corpus questions where the second search depends on the first's result — "how does our refund policy differ from the returns policy?" |
| **Vision** (`vision_agent.py`) | `inspect_image`, `read_text`, `search_knowledge_base`, `finish` | Reading an image by deciding *how* to read it |

A sub-agent is just a `Tool` whose `run` happens to be another loop — `Tool.run: Callable[[dict], str]` was already the whole interface delegation needed, and the loop did not change to support it.

### Why the vision agent decides rather than pipelines

OCR and a vision-language model are good at opposite things. Tesseract is exact on printed characters, free, local, and completely blind to meaning — nothing on diagrams, charts or handwriting. A vision model understands layout, charts and what a document *is*, and misreads long digit strings, which is exactly what an invoice total is.

So "what is the total on this receipt?" needs both: the vision model to find where the total is, character recognition to read the digits correctly. Which combination an image needs depends on what's in it, and the only way to know is to look — an agent's decision, not a config flag.

Measured over 7 labelled cases (`evals/`), each also run through a single `POST /vision/analyze` call as the baseline:

| | Passed | p50 latency | Median steps |
| --- | --- | --- | --- |
| single vision call (baseline) | 5 / 7 | 26.4 s | — |
| vision agent | **6 / 7** | 30.6 s | 3 |

Read that as a signal, not a benchmark, and read the margin sceptically. The set is small and the images are drawn rather than photographed (`tests/image_builder.py`), so recognition does better here than it would on real input. More to the point, one of the two baseline failures was a provider rate-limit rather than a wrong answer — on a case the agent also failed — so the honest reading is that the gap rests on one case.

That one case is the interesting one: a blank page, which the agent declined and the baseline described anyway. The agent's own failure is instructive in the other direction — a low-resolution total read as `84.80` instead of `84.50`, attributed in the answer to `read_text`, so the trace says exactly where the error entered.

> **These numbers were measured on `llama-3.3-70b-versatile`, which Groq retired in August 2026.** The default is now `openai/gpt-oss-120b`. Every prompt in `agents/` was tuned against the old model, so the figures above — and the agentic-RAG ones — describe a model this project no longer runs. Re-run `python -m evals.runner` before quoting them. Retrieval-only measurements (reranking, fusion, hybrid) are unaffected: nothing in that path calls an LLM.

Accuracy alone can't tell an agent that *decided* how to read an image from one that runs every tool on everything, so the cases assert on tools too: which the run must have called, and which it must **not** have.

### The three things a tree gets silently wrong

All three are corruption rather than crashes, which is why each has a named mechanism:

**Budget.** A 6-step supervisor free to call a 6-step specialist every step has a worst case of 36 LLM calls. `StepBudget` is one pool shared by reference across the whole tree, and every loop stops at whichever bites first — the pool or its own ceiling. `SUPERVISOR_TREE_BUDGET` (default 14) is what actually bounds a delegating run.

**Citation labels.** Two specialists each numbering from their own ledger both hand out `[E1]` for different passages; merged, the citation list points at the wrong text with nothing raised anywhere. One `EvidenceLedger`, owned by the supervisor, passed to every specialist.

**The trace.** A delegation that returned only its answer would collapse four steps into one string — and the trace is the only way a reader can check an answer assembled out of sight. Sub-steps arrive as `children` on the step that caused them, nested arbitrarily deep, and stream live rather than after.

### The critic

`SUPERVISOR_CRITIC_ENABLED=true` (default) reviews the draft against what the specialists actually reported, before the user sees it. It is a **gate on `finish`**, not a fourth tool: a reviewer the supervisor could choose to call would be skipped on exactly the answers that most need one — the confident ones.

It needed no new control flow. `ActionRejected` already meant "the tool declined, here is what to do instead", and the loop already turns that into an observation costing one step, so an objection is fed back as the next observation and the answer is rewritten once.

Two deliberate properties. It **fails open** — an unparseable verdict, an empty reply or a provider error all approve, because refusing to deliver finished, probably-correct work because a *second* model couldn't produce JSON turns a degradation into an outage. And it **doesn't run at all** when nothing was gathered: "is this supported by the evidence?" is meaningless for a question answered from the model's own knowledge, so a directly-answered question costs exactly what it did before the critic existed.

### Checking the numbers against the pixels

The vision agent's central promise — "never quote a number you have only seen through `inspect_image`" — was a prompt, and a model that ignores a prompt produces an answer indistinguishable from one that followed it. Meanwhile the OCR grid is sitting in memory, which is exactly what the claim can be checked against.

`agents/value_check.py` reports rather than enforces, and the worked example says why:

> "£84.50 across 2 covers is £42.25 a head, inside the £50 cap [E1]."

`84.50` is on the receipt. `42.25` is arithmetic the agent did correctly, and `50` came from a retrieved policy passage. An enforcing validator would flag two correct values and reject the right answer. So it returns a signal — `unverified_values` — answering "which numbers here did character recognition not confirm?" and leaving the judgement to the reader, who has the trace.

Matching is forgiving on form and strict on digits: both sides reduce to digit sequences, so `£1,234.50` matches an OCR grid reading `1234.50`. Comparison is per-token rather than against one concatenation of the page — otherwise a policy limit of `£50` quoted from a document matches the `50` inside the receipt's own `84.50`, and the check blesses a figure that never came from the image.

### Observability

One structured log line per run (`agents/run_log.py`), each carrying the id of the run that caused it. A response body can't answer "which layer spent the steps — the supervisor, or one specialist it called four times?", because that's a question about many runs rather than about this one. Parentage is tracked through a `contextvars.ContextVar` rather than a threaded parameter: sub-agents are built at dependency-injection time and invoked deep inside a tool's `run`, so there is no call signature in between that could carry a parent id.

---

## Agent2Agent (A2A)

`A2A_ENABLED=true` exposes the research agent to *other* agents over the [A2A protocol](https://a2a-protocol.org):

| Route | Purpose |
| --- | --- |
| `GET /.well-known/agent-card.json` | Discovery — what this agent can do, written for a model rather than a developer |
| `POST /a2a/v1` | JSON-RPC: `message/send`, `tasks/get` |

The card is served at the site root, deliberately not under `/api/v1` — the path is part of the protocol, the same way `/.well-known/openid-configuration` is. A client that has to be told where the card lives has not discovered anything.

The mapping is small: message text is the question, `contextId` is a `conversation_id`, and the run comes back as three artifacts — `answer`, `evidence`, `trace`.

Evidence is a separate artifact with **explicit `[E#]` labels next to the `chunkId` each stands for**, and that is the part worth reading twice. In-process, one `EvidenceLedger` shared by reference makes `[E3]` mean one passage everywhere. Over HTTP there is no shared object: this process numbers from E1, the caller numbers from E1, and merging two peers' answers silently points citations at the wrong text. Nothing here can fix that alone — only the caller knows what else it is merging — but emitting the correspondence explicitly, rather than leaving it implicit in list order, is what makes the fix possible.

`tasks/get` is backed by an in-process, bounded task store. That is a development-only answer and `a2a/task_store.py` says so: with more than one worker, a task created on A isn't found on B, and a restart tells a retrying caller its work never happened. The interface is the seam — a Redis implementation is the same three methods.

> **Nothing here checks credentials.** Any caller who can reach the port can spend this deployment's LLM budget and read answers drawn from its corpus. Turn it off, or put authentication in front of it, anywhere the port is reachable.

---

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

http://localhost:3000 — chat at `/`, and the document reader at `/read`.

`/read` is its own route rather than a mode inside the chat: the interaction is one image and one question rather than a conversation, and the output is a trace plus an answer plus caveats — a shape a chat bubble would have to be contorted to hold. The chat's own image path stays on `/vision/analyze`, which is still the right call for "describe this photo": one round trip instead of five.

The trace stays on screen after the run finishes. A run takes several seconds per step and produces no prose until the last one, so without it the user watches a spinner — and an answer assembled over four tool calls they never asked for isn't something they can sanity-check any other way.

### Tests

```bash
cd backend && .venv/bin/pytest       # backend
cd frontend && npx tsc --noEmit && npx eslint .   # frontend
```

The vision-agent evaluation is **not** part of pytest — it needs a live API key, spends provider quota on every run, and its results are measurements rather than assertions, so a number that moved is information rather than a build failure. Run it deliberately:

```bash
cd backend && .venv/bin/python -m evals.runner            # every case, with baseline
.venv/bin/python -m evals.runner --case clean-total       # one case
.venv/bin/python -m evals.runner --no-baseline            # skip the /vision/analyze comparison
```

Reports land in `backend/evals/reports/`. The graders themselves are unit-tested (`tests/test_eval_grading.py`), so the scoring can be trusted without spending anything to check it.

---

## API

All routes are under `/api/v1`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness — deliberately never cached |
| `POST` | `/chat` | Chat with conversation memory |
| `POST` | `/chat/stream` | Same, streamed token by token over SSE |
| `GET` | `/chat/{conversation_id}/history` | Full stored history for a conversation |
| `POST` | `/agent/ask` | Supervisor — answers directly or delegates, with the trace that produced it |
| `POST` | `/agent/ask/stream` | Same, steps streamed over SSE (this is what the UI uses) |
| `POST` | `/documents/upload` | Upload a document (PDF, DOCX, HTML, Markdown, CSV, TXT) |
| `POST` | `/documents/ingest` | Chunk, embed, and index a document |
| `GET` | `/documents` | List ingested documents |
| `POST` | `/documents/search` | Raw similarity search over chunks, filterable by author/title/date (dense only) |
| `POST` | `/rag/ask` | Retrieval-augmented answer, with per-source retrieval scores |
| `POST` | `/rag/chat` | RAG with conversation memory |
| `POST` | `/research/ask` | Multi-hop research over the corpus — trace, evidence, and why it stopped |
| `POST` | `/vision/analyze` | Describe or answer questions about an image (one round trip) |
| `POST` | `/vision/ask` | Vision & OCR agent — decides how to read the image, may consult the corpus |
| `POST` | `/vision/ask/stream` | Same, steps streamed over SSE (this is what `/read` uses) |
| `POST` | `/audio/analyze` | Transcribe and reason over audio |
| `POST` | `/stream/frame` | Submit one sampled frame from a live stream |
| `POST` | `/stream/{session_id}/end` | Tear down a stream session |

Two routes sit outside `/api/v1`, at the site root, because the A2A protocol fixes their location: `GET /.well-known/agent-card.json` and `POST /a2a/v1`. Both disappear entirely when `A2A_ENABLED=false`.

`/agent/ask/stream` sends steps rather than token deltas. The old router forwarded one tool's generation token by token; a supervisor's answer is already whole inside its `finish` action by the time the loop sees it, so what's worth streaming is the reasoning that produced it:

```
{"type": "start",   "conversation_id": "..."}     exactly one, first
{"type": "step",    "index": 1, "depth": 0, ...}  one per completed turn
{"type": "tool",    "tool": "research_documents"} exactly one
{"type": "sources", "sources": [...]}             at most one, if any
{"type": "answer",  "content": "..."}             exactly one
{"type": "done",    "stopped_because": "..."}     terminates a good stream
{"type": "error",   "detail": "..."}              terminates a bad one
```

---

## Memory

Two independent layers, deliberately not synchronized:

**Short-term (backend)** — `memory/conversation_memory.py`. SQLite, keyed by `conversation_id`. Every turn is stored permanently; only the most recent `CONVERSATION_HISTORY_LIMIT` (default 10) are replayed into the prompt. The cap is a view limit, not a retention limit.

**Session context (backend)** — an in-process ring buffer of recent stream observations per session, bounded and TTL-expired. Volatile by design.

There is no long-term memory: nothing is keyed by user, and nothing is summarized or promoted across conversations.

### Compaction

What used to happen to a turn that fell out of the window was *nothing*. Message 11 was still on disk, still rendered in the transcript the UI shows, and permanently invisible to the model — which produces a specific and confusing failure: the user can read, on screen, the message the assistant just contradicted.

`CONVERSATION_COMPACTION_ENABLED=true` (default) summarises the turns that drop out and sends the summary with every subsequent prompt:

```
[summary of everything up to message N]   ← one system message
message N+1 … message N+k                 ← the window, verbatim
```

Two markers make the split exact: the summary's `covered_through_id` high-water mark, and the oldest message still being sent verbatim. Anything strictly between them is uncompacted history, and that's what a pass consumes — so no message is in both halves or in neither. A pass rewrites the whole summary rather than appending to it, because a chain of summaries compounds its own distortions.

It runs after the assistant's turn is stored, once `CONVERSATION_COMPACTION_TRIGGER` messages have built up in the gap — so roughly one exchange in every trigger/2 pays an extra LLM call. That's a compromise, honestly: compacting *before* building the prompt would delay the answer the user is waiting on, and a background worker needs a queue this project doesn't have. `compact` never raises — a provider timeout during summarisation must not fail a turn that already succeeded.

### Attachments

Conversation memory stores what was *said* about an image or a recording; `memory/attachment_store.py` stores the bytes, so a turn's `attachment_ref` still resolves after the response is sent. Without it, vision had a silent failure: a stored turn was text only, so a second upload in the same conversation left the previous answer sitting in history as though it were about the new one — "it's red" read back as context for a photograph of something blue.

Refs are content-addressed (SHA-256 + extension), which buys three things for free: re-uploading the same image writes nothing, the ref is verifiable, and there's no id allocator to lock or sequence. The cost is equally plain — nothing is ever deleted, because two conversations about the same image share a file and a delete would need reference counting.

Audio gets the same treatment plus the transcript, capped at `AUDIO_TRANSCRIPT_MEMORY_CHARS` (default 4000). Minutes of speech become text once, at real cost; throwing that away with the response meant paying it again for every later question about the same recording. The cap is the compromise — a full hour-long transcript would swamp every subsequent prompt, and the recording is kept, so the untruncated text is one re-transcription away.

Both vision and audio **write** to memory without reading from it: a `conversation_id` files the analysis, it doesn't change it.

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
  agents/       reasoning loop, supervisor, research + vision specialists,
                critic, value check, run log
  a2a/          Agent Card, JSON-RPC dispatch, task store
  ai/           LLM, vision, transcription services
  rag/          loaders → blocks → chunks → embeddings → Chroma → retrieval
    loaders/    one module per format (pdf, docx, html, markdown, csv)
  memory/       conversation memory, compaction, attachment store (SQLite)
  processors/   streaming frame sampling and session state
  prompts/      one module per agent/role
  evals/        labelled vision-agent cases, graders, runner, reports
  tests/        pytest suite
frontend/
  app/          App Router routes (/ chat, /read document reader)
  components/   UI, including the agent trace
  hooks/        feature hooks (chat, vision, vision agent, audio, streaming, …)
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
| `EMBEDDING_CACHE_ENABLED` | `true` | Remember what a text embedded to — nothing to invalidate, since vectors are deterministic |
| `SUPERVISOR_MAX_STEPS` | `6` | Steps the supervisor itself may take |
| `SUPERVISOR_TREE_BUDGET` | `14` | Steps for the **whole tree**, from one shared pool — this is what bounds a delegating run |
| `SUPERVISOR_CRITIC_ENABLED` | `true` | Review the draft against the evidence before it reaches the user |
| `RESEARCH_MAX_STEPS` / `VISION_AGENT_MAX_STEPS` | `6` / `5` | Per-specialist ceilings, under the tree budget |
| `AGENT_TEMPERATURE` | `0.0` | Sampling for replies that get *parsed* — noise there reads as a prompt bug |
| `CONVERSATION_COMPACTION_ENABLED` | `true` | Summarise turns that fall out of the history window |
| `CONVERSATION_COMPACTION_TRIGGER` | `10` | Messages that must accumulate outside the window before a pass runs |
| `AUDIO_TRANSCRIPT_MEMORY_CHARS` | `4000` | Transcript kept alongside the turn; the recording itself is kept in full |
| `A2A_ENABLED` | `true` | Expose the research agent over A2A — **unauthenticated**, see above |
| `A2A_PUBLIC_BASE_URL` | `http://localhost:8000` | What a *third party* can resolve, not what this process binds to |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `60` / `2` | Deliberately below the SDK's 600s default: these run in FastAPI's threadpool |
| `MAX_UPLOAD_SIZE_MB` | `25` | PDF upload ceiling |
| `STREAM_SAMPLING_INTERVAL_SECONDS` | `2.0` | One frame every N seconds |
| `CORS_ORIGINS` | `localhost:3000, localhost:8080` | Comma-separated |

Local data (`backend/data/`) — SQLite databases, the Chroma index, the embedding cache, stored attachments, and uploads — is gitignored and regenerated on first run.
