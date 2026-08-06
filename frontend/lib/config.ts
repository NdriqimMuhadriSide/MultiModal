/**
 * Centralized environment configuration.
 *
 * Nothing else in the codebase should read `process.env` directly for API
 * URLs — import from here instead, mirroring how the backend centralizes
 * env access in `app/core/config.py`. This keeps env access in one place
 * and makes it trivial to add validation later.
 *
 * NEXT_PUBLIC_API_URL should point at the *host* of the FastAPI app
 * (e.g. http://localhost:8000), without the `/api/v1` prefix — the prefix
 * is appended once here because it's an implementation detail of the
 * backend's routing, not something every call site should know about.
 */

const DEFAULT_API_URL = "http://localhost:8000";
const API_V1_PREFIX = "/api/v1";

/** Base host of the backend, e.g. "http://localhost:8000" */
export const API_URL: string =
  process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;

/** Base URL including the versioned API prefix, e.g. "http://localhost:8000/api/v1" */
export const API_V1_URL = `${API_URL.replace(/\/$/, "")}${API_V1_PREFIX}`;
