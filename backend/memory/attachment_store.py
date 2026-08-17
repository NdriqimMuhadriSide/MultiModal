"""
Attachment store - the bytes half of multimodal memory.

memory/conversation_memory.py stores what was *said* about an image or a
recording. This stores the image or the recording itself, so a turn's
`attachment_ref` points at something that still exists after the response
has been sent.

Without it the vision endpoint had a specific, silent failure: a stored
turn was text only, nothing recorded which picture it described, and a
second upload in the same conversation left the previous answer sitting in
history as though it were about the new one. "It's red" would be read back
as context for a photograph of something blue. The fix needs two halves -
a ref on the turn (conversation_memory) and a file the ref resolves to
(here) - because a ref alone can tell you two turns are about *different*
images but cannot give you back the earlier one.

CONTENT-ADDRESSED

The ref is the SHA-256 of the bytes plus an extension, and the file is
named after it. Three things fall out of that, all of which would
otherwise need code:

    - Re-uploading the same image is free. The digest matches, the file is
      already there, and nothing is written. A user who asks four questions
      about one photograph stores it once.
    - The ref is verifiable. Bytes that hash to something else are not the
      bytes that turn was about, whatever the filename says.
    - There is no id allocator, so nothing has to be locked or sequenced
      when two requests store at the same time.

The cost is the obvious one: nothing is ever deleted, and two conversations
about the same image share a file, so a delete would need reference
counting. Not built, because nothing deletes yet - and a store that grows
is a smaller problem than one that hands back a file a *different*
conversation is still using.

MIME TYPES

The extension is derived from the mime type on the way in and mapped back
on the way out. That round trip canonicalises rather than loses: the
non-standard types browsers actually send ("image/jpg", "audio/mp3") come
back as the real ones ("image/jpeg", "audio/mpeg"), and every value the
reverse map produces is one ai/vision_service.py and
processors/audio/audio_validator.py already accept - so a file loaded back
out of here can be sent straight to the model that rejected its original
spelling.

Local files under settings.attachment_dir, for the same reason the rest of
the project is local: no S3, no bucket, no credentials. Swapping to object
storage later means replacing this file's internals and keeping
store/load.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Mime type -> extension, for the formats the vision and audio paths accept.
# The several spellings of the same format are deliberate: browsers disagree
# (Safari says "audio/mp4" where Chrome says "audio/x-m4a"), and the upload
# is the one place we don't get to choose.
_EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
}

# Extension -> the *canonical* mime type for it. Not built by inverting
# _EXTENSIONS: that mapping is many-to-one, and inverting it would pick a
# winner by dict ordering - which would silently change the day someone
# reorders the entries above. Spelled out, so the choice is the file's.
_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}

# What a ref is allowed to look like. Every ref this module produces is a
# hex digest and a known extension, so anything else is either corruption
# or a caller passing user input straight through - and since a ref becomes
# a path, "../../etc/passwd" has to be unrepresentable rather than merely
# unlikely.
_REF_PATTERN = re.compile(r"^[0-9a-f]{64}\.[a-z0-9]{2,5}$")


@dataclass(frozen=True)
class Attachment:
    """A stored file, read back."""

    ref: str
    data: bytes
    mime_type: str


class UnsupportedAttachmentType(ValueError):
    """Raised when asked to store bytes whose mime type has no extension here."""


class AttachmentStore:
    """Content-addressed local file storage for uploaded images and audio."""

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def store(self, data: bytes, mime_type: str) -> str:
        """
        Persist `data` and return the ref that identifies it.

        Storing the same bytes twice returns the same ref and writes
        nothing the second time.

        Raises:
            ValueError: if `data` is empty.
            UnsupportedAttachmentType: if `mime_type` isn't one this store
                knows an extension for.
        """
        if not data:
            raise ValueError("data must not be empty.")

        extension = _EXTENSIONS.get((mime_type or "").lower().strip())
        if extension is None:
            raise UnsupportedAttachmentType(
                f"Cannot store an attachment of type '{mime_type or 'unknown'}'."
            )

        ref = f"{hashlib.sha256(data).hexdigest()}{extension}"
        path = self._root / ref
        if not path.exists():
            # Written to a temporary name and moved into place, so a crash
            # mid-write cannot leave a truncated file sitting at the name a
            # ref resolves to - which would be worse than no file at all,
            # since the digest says the content is intact. os.replace is
            # atomic within a filesystem, and both paths are in _root.
            temporary = self._root / f"{ref}.partial"
            temporary.write_bytes(data)
            temporary.replace(path)

        return ref

    def load(self, ref: str) -> Attachment | None:
        """
        Read back the attachment `ref` identifies, or None if there isn't
        one - a ref from a conversation whose files have been cleared, or a
        database copied without its attachment directory.

        None rather than an exception because a missing attachment is a
        recoverable state for every caller: the vision endpoint asks the
        user to re-upload, which is what it did for every request before
        this store existed.
        """
        if not ref or not _REF_PATTERN.match(ref):
            return None

        path = self._root / ref
        if not path.is_file():
            return None

        mime_type = _MIME_TYPES.get(path.suffix)
        if mime_type is None:
            return None

        return Attachment(ref=ref, data=path.read_bytes(), mime_type=mime_type)


_store_instance: AttachmentStore | None = None


def get_attachment_store() -> AttachmentStore:
    """
    Return a process-wide AttachmentStore.

    Same shape as memory/conversation_memory.py's get_conversation_memory -
    a module global rather than @lru_cache, so a test can point this at a
    tmp_path without a cached instance outliving it.
    """
    global _store_instance
    if _store_instance is None:
        from app.core.config import settings

        _store_instance = AttachmentStore(root_dir=settings.attachment_dir)
    return _store_instance
