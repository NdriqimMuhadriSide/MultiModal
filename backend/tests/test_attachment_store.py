import hashlib

import pytest

from memory.attachment_store import (
    AttachmentStore,
    UnsupportedAttachmentType,
)


@pytest.fixture
def store(tmp_path):
    return AttachmentStore(root_dir=str(tmp_path / "attachments"))


def test_store_returns_a_content_addressed_ref(store):
    ref = store.store(b"fake png bytes", mime_type="image/png")

    assert ref == f"{hashlib.sha256(b'fake png bytes').hexdigest()}.png"


def test_round_trip_returns_the_same_bytes(store):
    ref = store.store(b"fake jpeg bytes", mime_type="image/jpeg")

    attachment = store.load(ref)

    assert attachment is not None
    assert attachment.data == b"fake jpeg bytes"
    assert attachment.ref == ref


def test_storing_the_same_bytes_twice_is_one_file(store, tmp_path):
    first = store.store(b"same image", mime_type="image/png")
    second = store.store(b"same image", mime_type="image/png")

    assert first == second
    assert len(list((tmp_path / "attachments").iterdir())) == 1


def test_different_bytes_get_different_refs(store):
    assert store.store(b"image a", mime_type="image/png") != store.store(
        b"image b", mime_type="image/png"
    )


def test_mime_type_is_canonicalised_on_the_way_back(store):
    # What a browser sends is not always a real mime type. The point of the
    # round trip is that what comes back is one the vision service accepts.
    assert store.load(store.store(b"x", mime_type="image/jpg")).mime_type == "image/jpeg"
    assert store.load(store.store(b"y", mime_type="audio/mp3")).mime_type == "audio/mpeg"
    assert store.load(store.store(b"z", mime_type="audio/x-m4a")).mime_type == "audio/mp4"


def test_store_rejects_an_unsupported_mime_type(store):
    with pytest.raises(UnsupportedAttachmentType):
        store.store(b"some bytes", mime_type="application/pdf")


def test_store_rejects_empty_data(store):
    with pytest.raises(ValueError):
        store.store(b"", mime_type="image/png")


def test_load_returns_none_for_an_unknown_ref(store):
    assert store.load(f"{'a' * 64}.png") is None


def test_load_returns_none_rather_than_escaping_the_store(store, tmp_path):
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"not yours")

    # A ref becomes a path, so this has to be unrepresentable rather than
    # merely unlikely - even though every ref today comes from our own
    # database.
    assert store.load("../secret.png") is None
    assert store.load("/etc/passwd") is None
    assert store.load("") is None


def test_load_ignores_a_partial_write(store, tmp_path):
    ref = store.store(b"real bytes", mime_type="image/png")
    (tmp_path / "attachments" / f"{ref}.partial").write_bytes(b"truncated")

    # The half-written file is not addressable, and the real one is intact.
    assert store.load(f"{ref}.partial") is None
    assert store.load(ref).data == b"real bytes"
