/**
 * useHealthCheck — runs the backend health check once on mount and exposes
 * a simple status for the UI (Header) to render.
 *
 * This hook owns the *orchestration* (calling the service, handling
 * loading/error, cleanup on unmount) so Header.tsx stays a pure
 * presentation component that just reads `status`.
 */
"use client";

import { useEffect, useState } from "react";
import { HealthService } from "@/services/health-service";
import type { BackendStatus } from "@/types";

export function useHealthCheck() {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [appName, setAppName] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    HealthService.check(controller.signal)
      .then((response) => {
        setStatus("online");
        setAppName(response.app_name);
      })
      .catch((error) => {
        // Ignore aborts caused by unmount/fast-refresh; anything else means
        // the backend is unreachable or returned an error.
        if (error?.name === "AbortError") return;
        setStatus("offline");
      });

    return () => controller.abort();
  }, []);

  return { status, appName };
}
