/**
 * Types for the backend health check contract.
 *
 * Mirrors `app/schemas/health.py::HealthResponse` on the backend so the
 * frontend and backend never drift out of sync silently — if the backend
 * shape changes, this is the one place to update.
 */

/** Raw JSON shape returned by GET /api/v1/health */
export interface HealthResponse {
  status: string;
  app_name: string;
  environment: string;
}

/**
 * UI-facing connection state. This is derived from the health check call
 * (or its failure) and drives the "Backend Connected / Offline" indicator.
 */
export type BackendStatus = "checking" | "online" | "offline";
