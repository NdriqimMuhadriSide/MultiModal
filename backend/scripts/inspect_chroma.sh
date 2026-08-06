#!/usr/bin/env bash
#
# Print the contents of the Chroma store as a readable table.
#
# The SQLite viewer extension shows the raw tables, but Chroma stores metadata
# key/value per row (one row per *field*, not per chunk) and keeps the chunk
# text under the key "chroma:document". This pivots that back into one row per
# chunk. Reads the file directly - the backend does not need to be running.
#
# Usage:
#   ./scripts/inspect_chroma.sh             # summary table
#   ./scripts/inspect_chroma.sh --full      # full chunk text
#   ./scripts/inspect_chroma.sh --vectors   # decode the embedding blobs
#
set -euo pipefail

BACKEND="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$BACKEND/data/chroma/chroma.sqlite3"

if [[ ! -f "$DB" ]]; then
  echo "No Chroma store at $DB (nothing ingested yet)." >&2
  exit 1
fi

# Vectors are packed float32 BLOBs in embeddings_queue (Chroma's write-ahead
# log) - 4 bytes per dimension, no separators, which is why a SQLite viewer
# renders them as mojibake. sqlite3 can't unpack them, so shell out to Python.
if [[ "${1:-}" == "--vectors" ]]; then
  "$BACKEND/.venv/bin/python" - "$DB" <<'PY'
import sqlite3, struct, sys

con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
rows = con.execute(
    "SELECT id, vector, encoding FROM embeddings_queue "
    "WHERE vector IS NOT NULL ORDER BY seq_id"
).fetchall()

# The log keeps deleted chunks too; operation 3 rows mark what is gone.
deleted = {r[0] for r in con.execute(
    "SELECT id FROM embeddings_queue WHERE operation = 3")}

for chunk_id, blob, encoding in rows:
    if encoding != "FLOAT32":
        print(f"{chunk_id}  (unhandled encoding {encoding})")
        continue
    vec = struct.unpack(f"<{len(blob) // 4}f", blob)
    head = ", ".join(f"{x:+.4f}" for x in vec[:6])
    tag = "  [DELETED]" if chunk_id in deleted else ""
    print(f"{chunk_id}{tag}\n  dims={len(vec)}  [{head}, ...]\n")

print(f"{len(rows)} vectors in the log, {len(deleted)} of them deleted")
PY
  exit 0
fi

# Chunk text is long; truncate unless --full was passed.
if [[ "${1:-}" == "--full" ]]; then
  TEXT="MAX(CASE WHEN m.key='chroma:document' THEN m.string_value END)"
else
  TEXT="substr(MAX(CASE WHEN m.key='chroma:document' THEN m.string_value END), 1, 70)"
fi

sqlite3 "file:$DB?mode=ro" <<SQL
.headers on
.mode column
.width 38 10 34

SELECT name AS collection, dimension AS dims,
       (SELECT count(*) FROM embeddings) AS chunks
FROM collections;

.print ''

.width 46 14 5 5 72
SELECT e.embedding_id AS chunk_id,
       MAX(CASE WHEN m.key='filename'    THEN m.string_value END) AS filename,
       MAX(CASE WHEN m.key='page_number' THEN m.int_value    END) AS page,
       MAX(CASE WHEN m.key='chunk_index' THEN m.int_value    END) AS chunk,
       $TEXT AS text
FROM embeddings e
JOIN embedding_metadata m ON m.id = e.id
GROUP BY e.id
ORDER BY filename, page, chunk;
SQL
