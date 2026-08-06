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

    # --- Chunking (Phase 4A document ingestion) ---
    # Character-based (not token-based) for simplicity - see rag/text_splitter.py
    # for the char->token rule-of-thumb reasoning. 800/150 is this project's
    # tuned default (smaller than the text_splitter module's own 1000/200
    # fallback) - overridable per environment without a code change.
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

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
