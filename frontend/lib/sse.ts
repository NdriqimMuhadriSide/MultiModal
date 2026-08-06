/**
 * Server-Sent Event reader for POST requests.
 *
 * Deliberately not built on `apiClient`: every method there ends in
 * `response.json()`, which reads the body to completion — the exact thing
 * streaming exists to avoid. This is the only place in the app that calls
 * `fetch` directly, and the reason is structural, not an oversight.
 *
 * `EventSource` isn't an option either, despite this being SSE: it only
 * issues GET requests with no body, and these endpoints take a JSON body.
 * So the framing is parsed by hand below — it's a small format.
 *
 * Shared by ChatService and AgentService, which consume the same wire
 * format from backend/app/api/v1/sse.py.
 */
import { ApiError } from "@/types/api";

/**
 * POSTs `body` and yields each parsed event as it arrives.
 *
 * Two distinct failure modes, and callers need to tell them apart:
 *   - The request never started (offline, or a non-2xx before any bytes).
 *     Throws `ApiError`, matching every other service, so the existing
 *     error-mapping and offline-queue paths keep working unchanged.
 *   - The provider died mid-answer. That arrives as an `error` *event*,
 *     because the response committed as a 200 before the model produced
 *     anything and a status code is no longer available.
 */
export async function* streamSse<TEvent>(
  url: string,
  body: unknown,
  signal?: AbortSignal
): AsyncGenerator<TEvent> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch {
    // fetch() itself threw — the request never reached the server. A null
    // status is what api-client uses for this, and what hooks already read
    // as "Backend Offline" (and what useAgent tests before queueing).
    throw new ApiError("Network request failed", null, url);
  }

  if (!response.ok || !response.body) {
    throw new ApiError(`Request failed with status ${response.status}`, response.status, url);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      // `stream: true` matters: a network chunk boundary can fall inside a
      // multi-byte character, and decoding it standalone would corrupt it.
      buffer += decoder.decode(value, { stream: true });

      // Frames are terminated by a blank line. A partial frame stays in the
      // buffer until the rest of it arrives.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const line = frame.trim();
        if (!line.startsWith("data: ")) continue;
        yield JSON.parse(line.slice("data: ".length)) as TEvent;
      }
    }
  } finally {
    // Covers an abort or an early `break` by the consumer: without this the
    // connection stays open and the backend keeps generating tokens for an
    // answer nobody is reading.
    await reader.cancel().catch(() => {});
  }
}
