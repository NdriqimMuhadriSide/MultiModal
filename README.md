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
   │  rag/         PDF loader → splitter → embeddings →        │
   │               ChromaDB → retriever → context builder     │
   │  agents/      LangGraph StateGraph (route → tool → answer)│
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
| `GET` | `/chat/{conversation_id}/history` | Full stored history for a conversation |
| `POST` | `/agent/ask` | LangGraph agent — routes to a tool, then answers |
| `POST` | `/documents/upload` | Upload a PDF |
| `POST` | `/documents/ingest` | Chunk, embed, and index a document |
| `GET` | `/documents` | List ingested documents |
| `POST` | `/documents/search` | Raw similarity search over chunks |
| `POST` | `/rag/ask` | Retrieval-augmented answer |
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
  rag/          PDF → chunks → embeddings → Chroma → retrieval
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
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Document chunking |
| `MAX_UPLOAD_SIZE_MB` | `25` | PDF upload ceiling |
| `STREAM_SAMPLING_INTERVAL_SECONDS` | `2.0` | One frame every N seconds |
| `CORS_ORIGINS` | `localhost:3000, localhost:8080` | Comma-separated |

Local data (`backend/data/`) — SQLite databases, the Chroma index, and uploads — is gitignored and regenerated on first run.
