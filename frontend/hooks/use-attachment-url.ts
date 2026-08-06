/**
 * useAttachmentUrl — the best available source for an attachment's media.
 *
 * Two sources, in priority order:
 *
 *   1. `attachment.url` — a live object URL created earlier in *this* page
 *      session by useVision/useAudio. Already valid, nothing to do.
 *   2. `attachment.cacheKey` — the Cache API entry, read back after a reload
 *      once the object URL above has been blanked by the store.
 *
 * Returns null when neither works, which is a normal outcome (no Cache API,
 * a failed write, or an eviction) and renders the same "media expired"
 * placeholder that existed before any of this.
 *
 * Ownership is the subtle part. The URL from source 1 belongs to the hook
 * that created it and is still referenced by the message, so revoking it here
 * would break the live view. Only URLs this hook mints from the cache get
 * revoked — hence the `objectUrl` local, which is set solely on that path.
 */
"use client";

import { useEffect, useState } from "react";
import { getMediaObjectUrl } from "@/lib/media-cache";
import type { Attachment } from "@/types";

export function useAttachmentUrl(attachment: Attachment): string | null {
  const { url, cacheKey } = attachment;
  const [restored, setRestored] = useState<string | null>(null);

  useEffect(() => {
    if (url || !cacheKey) {
      setRestored(null);
      return;
    }

    let objectUrl: string | null = null;
    let cancelled = false;

    void getMediaObjectUrl(cacheKey).then((next) => {
      if (!next) return;
      // Unmounted (or the attachment changed) while the read was in flight.
      // The URL was still created, so it still has to be released.
      if (cancelled) {
        URL.revokeObjectURL(next);
        return;
      }
      objectUrl = next;
      setRestored(next);
    });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url, cacheKey]);

  return url || restored;
}
