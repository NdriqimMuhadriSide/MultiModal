# syntax=docker/dockerfile:1

# Python 3.12, not 3.13: a transitive `tokenizers` dependency has no prebuilt
# wheel for 3.13 and fails to build from source. The same constraint the
# README states for local development applies here.
FROM python:3.12-slim

# Tesseract is the OCR engine pytesseract wraps but deliberately does not
# bundle (rag/ocr.py). Without the binary, scanned PDFs are recorded as
# FAILED with a reason saying so, and the vision agent's `read_text` tool has
# nothing behind it - which is half of what that agent exists to decide
# between.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch from PyTorch's CPU-only index, and BEFORE requirements.txt.
#
# sentence-transformers pulls in torch, and the default PyPI wheel bundles the
# CUDA runtime - roughly 2.5GB of GPU libraries, on a server that has no GPU.
# Installing the CPU build first means pip sees the requirement already
# satisfied when it reads requirements.txt and leaves it alone. That is the
# difference between an image that is comfortably under 1GB and one that is
# too big for most platforms' build steps.
#
# The pin tracks what the project is developed against. If this ever fails
# with "no matching distribution", drop the `==` and take the current CPU
# build - sentence-transformers only requires torch>=1.11.
#
# The `rm` belongs in THIS layer, not a later one. Image layers are additive:
# a file deleted by a subsequent RUN is hidden from the final filesystem but
# its bytes are still carried in the layer that created it, so the image never
# shrinks. What goes is what only a developer *of torch* needs - its own test
# suite (90MB) and the C++ headers for compiling extensions against it (63MB).
#
# `torch/bin` is deliberately kept: it holds torch_shm_manager, which a
# DataLoader with worker processes shells out to, and 48MB is not worth the
# class of bug where inference works until the batch is large enough.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/test \
              /usr/local/lib/python3.12/site-packages/torch/include

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models into the image rather than fetching them at runtime.
#
# The embedding model loads at startup, but the cross-encoder loads LAZILY
# (rag/reranker.py `_load`) - so without this, ~80MB is downloaded from
# HuggingFace inside the first user's question, which then takes a minute and
# may well time out. Nothing about that failure points at its cause.
#
# HF_HOME is set as an ENV rather than only for the build step, because the
# runtime has to look in the same place or the baked cache is never found.
ENV HF_HOME=/opt/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Set AFTER the download above, or there would be nothing to cache.
#
# Without this the loaders still work, but they ask huggingface.co whether the
# cached files are current before using them - and the cross-encoder loads
# lazily, so that round trip happens inside the first user's question. Worse,
# when HF is unreachable it is five retries with backoff before the fallback
# to cache, turning an upstream outage into a ~30s stall on a request that had
# everything it needed on local disk the whole time. Measured, not assumed.
#
# The trade: pointing EMBEDDING_MODEL_NAME or RERANK_MODEL_NAME at a model
# that is not in the image now fails loudly instead of downloading it. That is
# the right failure for an image whose models are supposed to be fixed at
# build time - but it does mean changing either setting requires a rebuild.
ENV HF_HUB_OFFLINE=1

COPY . .

# ./data is where the persistent volume gets mounted: Chroma's index, the
# three SQLite databases, and the attachment store. Created here so the
# container still starts with no volume attached - nothing persists across a
# restart in that case, but nothing crashes either, which makes a
# misconfigured mount visibly wrong rather than fatal.
RUN mkdir -p data/chroma data/attachments data/uploads

# Runs as root, deliberately, and this is the first thing to tighten later.
# Mounted volumes arrive owned by root on both Fly and Railway, so a non-root
# user needs an entrypoint that fixes ownership before dropping privileges.
# That is worth doing - it is just not worth debugging on a first deploy.

EXPOSE 8000

# A shell is needed so ${PORT} expands - Fly and Railway both inject it and
# expect the app to bind it, and the fallback keeps `docker run -p 8000:8000`
# working locally. --host 0.0.0.0 is required too: binding localhost inside a
# container means nothing outside the container can reach it.
#
# The `exec` is the part that is easy to leave out and wrong to. Without it
# /bin/sh stays as PID 1 and does not forward signals, so the SIGTERM a
# platform sends before replacing a machine never reaches uvicorn - it is
# SIGKILLed seconds later instead. That drops in-flight SSE streams and
# interrupts SQLite mid-write on every single deploy. With `exec`, uvicorn
# replaces the shell and receives the signal itself.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
