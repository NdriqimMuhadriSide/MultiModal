"""
Centralized application configuration.

All environment-driven settings live here. Nothing else in the codebase
should read `os.environ` directly - import `settings` from this module
instead so configuration stays in one place and is easy to test/override.
"""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    app_name: str = Field(default="Multimodal AI Assistant", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")

    # --- CORS ---
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        alias="CORS_ORIGINS",
    )

    # --- OpenAI (kept for when billing is set up / vision features land) ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_vision_model: str = Field(default="gpt-4o", alias="OPENAI_VISION_MODEL")

    # --- Groq (current active provider for /chat and /vision - free tier, OpenAI-compatible API) ---
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_vision_model: str = Field(default="qwen/qwen3.6-27b", alias="GROQ_VISION_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    # --- Vector DB / RAG (used by future tasks) ---
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="documents", alias="CHROMA_COLLECTION_NAME")

    # --- Embeddings (local, free - runs on-device via sentence-transformers) ---
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )

    # --- Document registry (Phase 4A document ingestion - local SQLite, no extra infra) ---
    document_registry_db_path: str = Field(
        default="./data/documents.sqlite3", alias="DOCUMENT_REGISTRY_DB_PATH"
    )

    # --- PDF parsing (Phase 4A document ingestion) ---
    # Layout-aware extraction: read each page as a character grid and recover
    # reading order, tables, and running headers/footers (see rag/layout.py).
    # Costs a little more time per page than pypdf's default content-stream
    # order; turn it off to compare the two on a document, or if a particular
    # PDF's geometry confuses the segmenter.
    pdf_layout_mode: bool = Field(default=True, alias="PDF_LAYOUT_MODE")

    # --- OCR fallback for scanned PDFs (see rag/ocr.py) ---
    # Applies per page, only to pages that came back with (almost) no text, so
    # a document that mixes digital pages with scanned inserts pays the cost
    # for the scans alone. Requires the Tesseract binary; when it's missing,
    # ingestion carries on and the document is FAILED with the reason.
    ocr_enabled: bool = Field(default=True, alias="OCR_ENABLED")
    ocr_language: str = Field(default="eng", alias="OCR_LANGUAGE")
    # 300dpi is the usual floor for reliable OCR of body text; lower loses
    # small type, higher mostly buys render time.
    ocr_dpi: int = Field(default=300, alias="OCR_DPI")
    # A page with fewer than this many non-whitespace characters is treated as
    # having no text layer. Not zero: a scanned page often still carries a
    # stamped page number or a digital header over the image.
    ocr_min_text_chars: int = Field(default=32, alias="OCR_MIN_TEXT_CHARS")
    # OCR costs roughly a second per page and /documents/upload answers
    # synchronously, so a 500-page scan would hold the request open until it
    # timed out. Pages past this limit are left empty and the document says so.
    ocr_max_pages: int = Field(default=50, alias="OCR_MAX_PAGES")

    # --- Chunking (Phase 4A document ingestion) ---
    # Character-based (not token-based) for simplicity - see rag/text_splitter.py
    # for the char->token rule-of-thumb reasoning. 800/150 is this project's
    # tuned default (smaller than the text_splitter module's own 1000/200
    # fallback) - overridable per environment without a code change.
    # Sized in *tokens*, measured with the embedding model's own tokenizer -
    # not characters. The model has a hard token limit (256 for MiniLM-L6) and
    # crossing it is silent: it truncates and returns a vector describing only
    # the start of the chunk. Characters don't predict tokens well enough to
    # stay under that - 800 characters is ~180 tokens of prose but ~340 of
    # Markdown table rows - so the unit has to be the one the limit is in.
    #
    # The env vars are named ..._TOKENS rather than the old CHUNK_SIZE /
    # CHUNK_OVERLAP so a stale character-based value in an existing .env is
    # ignored instead of being silently reinterpreted as a token count.
    #
    # 256 is MiniLM's whole window, deliberately: chunks are clamped to the
    # model's real limit at ingestion time, and room for the section header is
    # reserved out of this budget rather than added on top of it.
    chunk_size_tokens: int = Field(default=256, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=48, alias="CHUNK_OVERLAP_TOKENS")
    # Prefix each chunk with the section it came from ("[2. Methods > 2.1
    # Field Sampling]"), so the text that gets embedded carries the context
    # the heading provided - see rag/structure.py. The prefix is added after
    # splitting, so a chunk can exceed chunk_size by the length of its header.
    chunk_section_headers: bool = Field(default=True, alias="CHUNK_SECTION_HEADERS")

    # --- Chunking strategy (see rag/chunking/__init__.py) ---
    # recursive | semantic | sentence_window | parent_document | propositional
    # `recursive` is the default because it is the only one with no runtime
    # dependency on a model: no embeddings at split time, no API key, no
    # per-chunk LLM call. The others beat it on particular documents, not in
    # general, and each costs something specific - see the module docstring.
    chunking_strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")

    # Cut where the distance between consecutive sentences is in the top N%
    # for that passage. A percentile rather than a fixed distance so the same
    # setting works on a rambling blog post and a tightly-argued contract.
    semantic_breakpoint_percentile: int = Field(
        default=95, ge=50, le=99, alias="SEMANTIC_BREAKPOINT_PERCENTILE"
    )
    # Sentences of context returned on each side of the embedded one.
    sentence_window_size: int = Field(default=3, ge=0, le=10, alias="SENTENCE_WINDOW_SIZE")
    # Size of the small children that get embedded on a parent's behalf.
    parent_child_tokens: int = Field(default=64, ge=8, alias="PARENT_CHILD_TOKENS")

    # --- Contextual retrieval (Anthropic's technique) ---
    # Off by default: it costs one LLM call per chunk at ingestion, which is by
    # far the most expensive thing in the pipeline and needs a working
    # GROQ_API_KEY. Composes with any chunking_strategy above.
    contextual_retrieval: bool = Field(default=False, alias="CONTEXTUAL_RETRIEVAL")
    # Reserved out of the chunk budget for the generated line, and the ceiling
    # the line is clamped to. Both, because the line is part of the embedded
    # text: unreserved it would push chunks past the model's limit, and
    # unclamped a model asked for one sentence sometimes writes four.
    contextual_reserved_tokens: int = Field(
        default=48, ge=8, alias="CONTEXTUAL_RESERVED_TOKENS"
    )

    # --- Retrieval ---
    # Several strategies point many chunks at one passage (a parent, a
    # sentence window). Retrieval fetches this multiple of top_k so that
    # collapsing duplicates still leaves top_k distinct passages.
    retrieval_overfetch: int = Field(default=4, ge=1, le=20, alias="RETRIEVAL_OVERFETCH")

    # dense | keyword | hybrid - see rag/retriever.py.
    #
    # `hybrid` is the default because the two halves fail in opposite
    # directions: embeddings match meaning but lose rare literal tokens (an
    # error code, a regulation number, a surname), BM25 matches those exactly
    # but returns nothing when question and document use different words. Set
    # to `dense` to get the pre-hybrid behaviour back verbatim.
    #
    # The keyword half costs no API key and no model: it is an inverted index
    # built in memory from what is already in Chroma (rag/keyword_index.py).
    # What it does cost is a rebuild on the first query after any ingest or
    # delete, proportional to corpus size.
    retrieval_mode: str = Field(default="hybrid", alias="RETRIEVAL_MODE")

    # How much each half counts for when their rankings are fused. Equal by
    # default; raise the keyword weight for identifier-heavy technical corpora,
    # raise the dense weight for prose where questions rarely reuse the
    # document's own vocabulary. Setting one to 0 disables that half, which is
    # the same thing as picking the other mode outright.
    hybrid_dense_weight: float = Field(default=1.0, ge=0.0, alias="HYBRID_DENSE_WEIGHT")
    hybrid_keyword_weight: float = Field(default=1.0, ge=0.0, alias="HYBRID_KEYWORD_WEIGHT")

    # Reciprocal rank fusion's smoothing constant (rag/fusion.py). 60 is the
    # published default: large relative to the ranks in play, so agreement
    # between the two retrievers outweighs either one's first place.
    rrf_k: int = Field(default=60, ge=1, alias="RRF_K")

    # BM25's term-frequency saturation (k1) and length normalisation (b).
    # 1.2 / 0.75 are the values Lucene ships; there is rarely a reason to
    # change them without a labelled evaluation set to change them against.
    bm25_k1: float = Field(default=1.2, ge=0.0, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0, alias="BM25_B")

    # --- Reranking (see rag/reranker.py) ---
    # A second scoring pass over the shortlist retrieval produced, using a
    # cross-encoder that reads the question and the chunk *together* - which
    # the embedding model cannot do, since chunk vectors are computed at
    # ingestion, before any question exists.
    #
    # On by default: it is local and free (no API key), and it is the largest
    # single retrieval-quality win available here. What it costs is a ~80MB
    # model downloaded on first use and a forward pass per candidate at query
    # time - so it is capped by the shortlist, never run over the corpus.
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL_NAME"
    )
    # Drop candidates the cross-encoder scores below this, in (0, 1) after the
    # sigmoid in rag/reranker.py. 0.0 keeps everything and only reorders,
    # which is the safe default: the reranker is much better at "which of
    # these is best" than at "is any of these good enough", and a floor set
    # too high turns answerable questions into "I don't know".
    rerank_min_score: float = Field(default=0.0, ge=0.0, le=1.0, alias="RERANK_MIN_SCORE")

    # --- Multi-query retrieval (see rag/query_expansion.py) ---
    # Ask an LLM for several phrasings of the question, retrieve for each, and
    # fuse all the rankings. Widens what retrieval can find when the document
    # words an answer differently from the way the user asked.
    #
    # Off by default, unlike hybrid and reranking, and for a different reason
    # than either: this one puts an LLM round-trip on the critical path of
    # every question, before retrieval can even start. That is a different
    # order of cost from the reranker's ~15ms of local compute, it needs
    # GROQ_API_KEY, and it consumes free-tier rate limit on every question.
    # CONTEXTUAL_RETRIEVAL is off for the same reason.
    multi_query_enabled: bool = Field(default=False, alias="MULTI_QUERY_ENABLED")
    # How many *extra* phrasings to generate. The original question is always
    # retrieved for as well, so 3 here means four searches per half.
    multi_query_count: int = Field(default=3, ge=1, le=10, alias="MULTI_QUERY_COUNT")

    # --- Corrective RAG (see rag/grading.py) ---
    # Grade what retrieval produced before answering from it, and act on the
    # grade: answer, retry with rephrased queries, or refuse.
    #
    # This is the only thing in the pipeline that can decline to answer. Every
    # other setting makes retrieval better at finding something; this one
    # notices when what it found is not worth answering from, which no amount
    # of retrieval quality can do for a question the corpus simply has no
    # answer to.
    #
    # Requires RERANK_ENABLED (the grade is a cross-encoder score - see
    # rag/grading.py for why nothing else works). An LLM is optional: with
    # one, a poor grade triggers a rephrase-and-retry; without one, grading
    # still decides answer-or-refuse, which is where the measured benefit was.
    #
    # Off by default because it is a genuine trade, not a free win. On this
    # project's 34-question eval it took correct refusals from 10% to 70%
    # and cost 12.5% false refusals on questions that were answerable. Worth
    # it when users ask things the corpus does not cover; not worth it when
    # nearly every question has an answer.
    corrective_rag_enabled: bool = Field(default=False, alias="CORRECTIVE_RAG_ENABLED")
    # At or above this, retrieval is trusted and the answer proceeds directly.
    corrective_accept_score: float = Field(
        default=0.5, ge=0.0, le=1.0, alias="CORRECTIVE_ACCEPT_SCORE"
    )
    # Below this, nothing retrieved is relevant and the pipeline refuses.
    # Between the two is the ambiguous band that triggers a retry.
    #
    # The default was chosen by sweeping it against the eval set rather than
    # by intuition, and it is far lower than a calibrated-probability reading
    # would suggest: the cross-encoder is an excellent ranker but a poorly
    # calibrated classifier, and some genuinely answerable questions have a
    # best score of 0.0001 even with the right chunk ranked first. Every
    # higher value traded away more answers than it bought refusals.
    corrective_reject_score: float = Field(
        default=0.001, ge=0.0, le=1.0, alias="CORRECTIVE_REJECT_SCORE"
    )

    # --- Agentic RAG ---
    # Rewrite a follow-up question into a standalone one before retrieving,
    # using the conversation so far (see rag/contextualizer.py).
    #
    # Retrieval has no memory: "how far ahead do I have to request it?"
    # reaches the index as six words, and the conversation that said what
    # "it" was never gets there. Passing history to the LLM at answer time -
    # which the chat path already does - cannot fix this, because by then
    # the chunks have already been chosen.
    #
    # On by default despite costing an LLM call, unlike the other
    # LLM-at-query-time features: it only runs on turns that have a
    # conversation behind them (the first question of every conversation
    # skips it entirely), and it fixes a defect rather than sharpening
    # something that already worked.
    query_contextualization_enabled: bool = Field(
        default=True, alias="QUERY_CONTEXTUALIZATION_ENABLED"
    )

    # When the knowledge-base tool retrieves nothing, let the agent take a
    # second hop and answer from the model's own knowledge, disclosing that
    # it did. Off means a mis-route to the knowledge base stays terminal:
    # the router's one guess is also its last.
    agent_kb_fallback_enabled: bool = Field(
        default=True, alias="AGENT_KB_FALLBACK_ENABLED"
    )

    # --- Conversation memory (local SQLite - no extra infra required) ---
    conversation_db_path: str = Field(
        default="./data/conversations.sqlite3", alias="CONVERSATION_DB_PATH"
    )
    conversation_history_limit: int = Field(
        default=10,
        alias="CONVERSATION_HISTORY_LIMIT",
        description="Max number of past messages (short-term memory window) sent to the LLM per request.",
    )

    # --- File uploads ---
    max_upload_size_mb: int = Field(default=25, alias="MAX_UPLOAD_SIZE_MB")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")

    # --- Audio understanding (Phase 5) ---
    # Whisper via Groq's OpenAI-compatible API (same GROQ_API_KEY/base URL as
    # chat/vision - see ai/transcription_service.py). whisper-large-v3 is
    # Groq's most accurate hosted Whisper model; whisper-large-v3-turbo is
    # faster/cheaper if latency matters more than accuracy for a given use case.
    groq_whisper_model: str = Field(default="whisper-large-v3", alias="GROQ_WHISPER_MODEL")
    # 25MB matches the Whisper API's own hard upload limit - enforced here,
    # ahead of the API call, so an oversized file fails fast with a clear
    # message instead of a confusing error from the transcription provider.
    max_audio_size_mb: int = Field(default=25, alias="MAX_AUDIO_SIZE_MB")

    # --- Real-time streaming (Phase 7) ---
    # A single sampled frame is a JPEG/PNG screenshot-sized image, nowhere
    # near a full video upload - a modest ceiling catches a misbehaving
    # client sending an oversized frame without needing the general
    # image/PDF/audio limits above.
    max_stream_frame_size_mb: int = Field(default=5, alias="MAX_STREAM_FRAME_SIZE_MB")
    # Sampling strategy: "interval" (every N seconds) or "frame_count"
    # (every Nth frame). See processors/streaming/frame_sampler.py for the
    # accuracy/latency/cost trade-off between the two.
    stream_sampling_strategy: str = Field(default="interval", alias="STREAM_SAMPLING_STRATEGY")
    # Default per the Phase 7 spec: "One frame every 2 seconds."
    stream_sampling_interval_seconds: float = Field(
        default=2.0, alias="STREAM_SAMPLING_INTERVAL_SECONDS"
    )
    # Only used when stream_sampling_strategy == "frame_count".
    stream_sampling_frame_count: int = Field(default=10, alias="STREAM_SAMPLING_FRAME_COUNT")
    # How many recent observations to keep per session for short-term
    # context (see processors/streaming/stream_processor.py's context
    # ring buffer) - bounded so long-running sessions don't grow memory
    # unboundedly; old observations age out once this many newer ones exist.
    stream_context_window_size: int = Field(default=10, alias="STREAM_CONTEXT_WINDOW_SIZE")
    # A session with no frames for this long is considered stale and
    # eligible for cleanup (see stream_processor.py's session expiry) -
    # bounds memory growth from abandoned sessions (e.g. a browser tab
    # closed without calling a stop endpoint).
    stream_session_ttl_seconds: int = Field(default=300, alias="STREAM_SESSION_TTL_SECONDS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()


settings = get_settings()
