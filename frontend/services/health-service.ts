/**
 * HealthService — talks to GET /health.
 *
 * Kept separate from ChatService because it belongs to a different backend
 * concern (liveness, not conversation). As more services are added
 * (VisionService, DocumentService, RagService, AgentService), each gets its
 * own file here so business/API logic never leaks into components.
 */
import { apiClient } from "@/lib/api-client";
import type { HealthResponse } from "@/types";

export const HealthService = {
  /**
   * Checks backend availability.
   *
   * The health route is mounted under the versioned router at
   * `/api/v1/health` (see backend/app/main.py, which includes
   * `api_router` with `settings.api_v1_prefix`, and
   * backend/app/api/v1/router.py which registers `health.router`).
   * apiClient defaults to the `/api/v1` base, so this resolves correctly.
   */
  check(signal?: AbortSignal): Promise<HealthResponse> {
    return apiClient.get<HealthResponse>("/health", { signal });
  },
};
