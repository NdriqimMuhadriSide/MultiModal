/**
 * Shared HTTP/transport-level types. These are generic to any endpoint,
 * as opposed to types/chat.ts and types/health.ts which describe specific
 * request/response payloads.
 */

/** Normalized error shape thrown by lib/api-client.ts for any failed request. */
export class ApiError extends Error {
  public readonly status: number | null;
  public readonly url: string;

  constructor(message: string, status: number | null, url: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
  }
}
