/**
 * ServiceWorkerRegistrar — installs public/sw.js. Renders nothing.
 *
 * Production only, and that's not caution — it's a hard requirement. The dev
 * server serves modules that change on every edit and pushes updates over
 * HMR; a worker sitting in front of those requests serves yesterday's code
 * and makes hot reload look broken in ways that are genuinely hard to
 * diagnose. Test offline behaviour with `npm run build && npm start`.
 *
 * Registration is deferred to the `load` event so it doesn't compete with the
 * page's own resources: the worker matters on the *next* visit, never this
 * one, so there's nothing to gain by fetching it early and real bandwidth to
 * lose on a slow connection.
 */
"use client";

import { useEffect } from "react";

export function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production") return;
    if (!("serviceWorker" in navigator)) return;

    const register = () => {
      // Failure is non-fatal by design — an unsupported browser, a blocked
      // registration, or an insecure origin all just mean no offline support.
      void navigator.serviceWorker.register("/sw.js").catch(() => {});
    };

    if (document.readyState === "complete") {
      register();
      return;
    }

    window.addEventListener("load", register);
    return () => window.removeEventListener("load", register);
  }, []);

  return null;
}
