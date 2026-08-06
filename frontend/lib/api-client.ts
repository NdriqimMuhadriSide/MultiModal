/**
 * Low-level HTTP client wrapping `fetch`.
 *
 * This is the single place that knows how to talk HTTP to the backend:
 * building URLs, setting headers, serializing bodies, parsing JSON, and
 * normalizing failures into `ApiError`. Feature services (e.g. ChatService)
 * sit on top of this and should never call `fetch` directly — that keeps
 * transport concerns (timeouts, headers, error shape) in one spot instead
 * of duplicated across every service.
 */
import { API_V1_URL } from "./config";
import { ApiError } from "@/types/api";

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
  /** Override the base URL for this call (defaults to the versioned API base). */
  baseUrl?: string;
}

async function request<TResponse>(
  path: string,
  options: RequestOptions = {}
): Promise<TResponse> {
  const { method = "GET", body, signal, baseUrl = API_V1_URL } = options;
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: {
        Accept: "application/json",
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (cause) {
    // Network-level failure: backend unreachable, DNS failure, CORS block, etc.
    throw new ApiError(
      cause instanceof Error ? cause.message : "Network request failed",
      null,
      url
    );
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new ApiError(detail, response.status, url);
  }

  // Some endpoints (e.g. future DELETE routes) may return no content.
  if (response.status === 204) {
    return undefined as TResponse;
  }

  return (await response.json()) as TResponse;
}

async function extractErrorDetail(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data?.detail === "string") return data.detail;
    return JSON.stringify(data);
  } catch {
    return response.statusText || `Request failed with status ${response.status}`;
  }
}

interface UploadOptions {
  signal?: AbortSignal;
  baseUrl?: string;
  /** Called with 0-100 as the browser reports upload progress. */
  onUploadProgress?: (percent: number) => void;
}

/**
 * Multipart/form-data POST, used for file uploads (e.g. /vision/analyze).
 *
 * Split from `request()` above because file uploads need two things plain
 * JSON requests don't: no `Content-Type` header (the browser sets the
 * multipart boundary itself) and, if progress reporting is requested,
 * `XMLHttpRequest` instead of `fetch` — `fetch` has no upload progress API.
 */
function postForm<TResponse>(
  path: string,
  formData: FormData,
  options: UploadOptions = {}
): Promise<TResponse> {
  const { signal, baseUrl = API_V1_URL, onUploadProgress } = options;
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;

  // No progress callback needed: plain fetch is simpler and sufficient.
  if (!onUploadProgress) {
    return (async () => {
      let response: Response;
      try {
        response = await fetch(url, { method: "POST", body: formData, signal });
      } catch (cause) {
        throw new ApiError(
          cause instanceof Error ? cause.message : "Network request failed",
          null,
          url
        );
      }
      if (!response.ok) {
        throw new ApiError(await extractErrorDetail(response), response.status, url);
      }
      return (await response.json()) as TResponse;
    })();
  }

  // Progress callback requested: use XHR, the only web API that exposes
  // upload progress events.
  return new Promise<TResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);

    if (signal) {
      if (signal.aborted) {
        reject(new ApiError("Request aborted", null, url));
        return;
      }
      signal.addEventListener("abort", () => xhr.abort());
    }

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onUploadProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      const isOk = xhr.status >= 200 && xhr.status < 300;
      let parsed: unknown = undefined;
      try {
        parsed = xhr.responseText ? JSON.parse(xhr.responseText) : undefined;
      } catch {
        parsed = undefined;
      }

      if (!isOk) {
        const detail =
          typeof (parsed as { detail?: string } | undefined)?.detail === "string"
            ? (parsed as { detail: string }).detail
            : xhr.statusText || `Request failed with status ${xhr.status}`;
        reject(new ApiError(detail, xhr.status, url));
        return;
      }
      resolve(parsed as TResponse);
    };

    xhr.onerror = () => reject(new ApiError("Network request failed", null, url));
    xhr.onabort = () => reject(new ApiError("Request aborted", null, url));

    xhr.send(formData);
  });
}

export const apiClient = {
  get: <TResponse>(path: string, options?: Omit<RequestOptions, "method" | "body">) =>
    request<TResponse>(path, { ...options, method: "GET" }),

  post: <TResponse, TBody = unknown>(
    path: string,
    body?: TBody,
    options?: Omit<RequestOptions, "method" | "body">
  ) => request<TResponse>(path, { ...options, method: "POST", body }),

  postForm: postForm,
};
