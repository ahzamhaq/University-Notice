"""One-off: fill notice_chunks.embedding_m3 with bge-m3 vectors (migration 002 -> 003).

Chunk text is already stored, so nothing is downloaded again. Resumable: only rows whose
embedding_m3 is still empty are processed. Uses EMBED_PROVIDER (cloudflare is much faster
than a CPU and gives identical vectors).

    EMBED_PROVIDER=cloudflare python scripts/reembed_bge_m3.py --dry-run   # cost estimate only
    EMBED_PROVIDER=cloudflare python scripts/reembed_bge_m3.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from embed_and_store import make_embedder, normalize_supabase_url  # noqa: E402
from scraper import setup_logging  # noqa: E402

log = logging.getLogger("reembed")
PAGE = 1000
BATCH = 50


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    setup_logging()
    load_dotenv(config.BASE_DIR / ".env")
    db = create_client(normalize_supabase_url(os.environ["SUPABASE_URL"]), os.environ["SUPABASE_KEY"])

    todo: list[dict] = []
    start = 0
    while True:
        rows = (
            db.table("notice_chunks").select("id,content").is_("embedding_m3", "null")
            .order("id").range(start, start + PAGE - 1).execute().data
        )
        todo += rows
        if len(rows) < PAGE:
            break
        start += PAGE

    tokens = sum(len(t) for t in tiktoken.get_encoding("cl100k_base").encode_batch([r["content"] for r in todo]))
    # bge-m3's tokenizer produces roughly 1.2x cl100k tokens on this text; Cloudflare bills
    # 1,075 neurons per million tokens and the free plan allows 10,000 neurons per day.
    log.info("%d chunks to embed, ~%.1fM tokens, ~%d Cloudflare neurons", len(todo), tokens / 1e6, tokens * 1.2 * 1075 / 1e6)

    if dry_run:
        return
    embedder = make_embedder()
    embedder.check()
    started, done = time.monotonic(), 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        vectors = embedder.embed_documents([r["content"] for r in batch])
        payload = [{"id": r["id"], "embedding": v} for r, v in zip(batch, vectors)]
        done += db.rpc("set_chunk_embeddings_m3", {"payload": payload}).execute().data
        if (i // BATCH) % 10 == 0 or done == len(todo):
            log.info("%d/%d chunks re-embedded (%.0fs)", done, len(todo), time.monotonic() - started)
    log.info("Done: %d chunks in %.0fs", done, time.monotonic() - started)


if __name__ == "__main__":
    main()
