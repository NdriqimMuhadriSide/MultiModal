/**
 * Client-side id generation.
 *
 * Everything the UI creates before the server has seen it — a chat, an
 * optimistic message, an attachment, a stream session — needs an id
 * immediately, so ids are minted here rather than waited on.
 *
 * `crypto.randomUUID()` is the obvious way to do that, and it is the way this
 * takes when it can. But it is marked `[SecureContext]` in the spec, which
 * means the browser only exposes it on `https://` and on `localhost`. Open the
 * same dev server from a phone on the LAN — `http://192.168.1.20:3000`, which
 * is the whole point of a mobile-first client — and `crypto` is still there
 * while `crypto.randomUUID` is `undefined`. The app then dies on the first
 * render that creates a chat, with "crypto.randomUUID is not a function".
 *
 * `crypto.getRandomValues()` carries no such gate, so the fallback formats a
 * v4 UUID out of 16 random bytes by hand. Same shape, same entropy, works on
 * plain http. `Math.random` sits behind that as a last resort for contexts
 * with no Web Crypto at all; these ids are local storage keys and React keys,
 * never secrets, so a weaker source degrades collision odds and nothing else.
 */

/** Formats 16 bytes as a canonical `8-4-4-4-12` hex string. */
function formatUuid(bytes: Uint8Array): string {
  // Stamp version (4) and variant (10xx) so the result is a well-formed v4.
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join("-");
}

/**
 * Returns a v4 UUID, by the best method this context actually offers.
 *
 * Use this everywhere instead of calling `crypto.randomUUID()` directly — a
 * direct call is a crash waiting for the first non-https visitor.
 */
export function newId(): string {
  if (typeof crypto !== "undefined") {
    if (typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
    if (typeof crypto.getRandomValues === "function") {
      return formatUuid(crypto.getRandomValues(new Uint8Array(16)));
    }
  }

  const bytes = new Uint8Array(16);
  for (let i = 0; i < bytes.length; i += 1) {
    bytes[i] = Math.floor(Math.random() * 256);
  }
  return formatUuid(bytes);
}
