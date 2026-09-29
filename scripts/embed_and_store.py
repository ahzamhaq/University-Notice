"""Ingest IPU notices: crawl -> download -> extract -> chunk -> embed (Ollama) -> Supabase.

Run it weekly. New PDFs are ingested, and stored notices older than REFRESH_DAYS are
re-downloaded; unchanged ones only get their update_date bumped, changed ones are
re-embedded.

    python scripts/embed_and_store.py                 # normal run
    python scripts/embed_and_store.py --max-pdfs 10   # quick test
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ollama
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from scraper import (  # noqa: E402
    NoTextError,
    NoticeCrawler,
    PdfLink,
    PoliteSession,
    ScrapeError,
    fetch_notice,
    setup_logging,
)

log = logging.getLogger("ingest")

PAGE_SIZE = 1000  # Supabase returns at most 1000 rows per request
INSERT_BATCH = 100
TITLE_ONLY_NOTE = (
    "(Only the title of this notice is available: the PDF is a scanned image. "
    "Open the PDF for the full details.)"
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: str) -> datetime:
    """Parse a Postgres timestamptz. Python 3.10's fromisoformat needs exactly 6 fraction digits."""
    value = value.replace("Z", "+00:00")
    value = re.sub(r"\.(\d+)", lambda m: "." + m.group(1).ljust(6, "0")[:6], value)
    return datetime.fromisoformat(value)


def normalize_supabase_url(url: str) -> str:
    """The client adds /rest/v1 itself; a URL copied with that suffix causes PGRST125 errors."""
    return re.sub(r"(/rest/v1)?/*$", "", url.strip())


@dataclass
class KnownNotice:
    id: int
    status: str
    content_hash: str | None
    updated_at: datetime


class NoticeStore:
    """Supabase access for the notices / notice_chunks tables (see supabase/schema.sql)."""

    def __init__(self, client: Client) -> None:
        self.db = client
        self.known: dict[str, KnownNotice] = {}
        self.cutoff = utcnow() - timedelta(days=config.REFRESH_DAYS)

    def load_index(self) -> None:
        """Cache every stored URL so duplicates are skipped without a query per PDF."""
        start = 0
        while True:
            rows = (
                self.db.table("notices")
                .select("id,url,status,content_hash,updated_at")
                .order("id")
                .range(start, start + PAGE_SIZE - 1)
                .execute()
                .data
            )
            for row in rows:
                self.known[row["url"]] = KnownNotice(
                    row["id"], row["status"], row["content_hash"], parse_ts(row["updated_at"])
                )
            if len(rows) < PAGE_SIZE:
                break
            start += PAGE_SIZE
        log.info("Loaded %d stored notice URLs from Supabase", len(self.known))

    def is_fresh(self, url: str) -> bool:
        """True if the URL is stored and was (re)processed within REFRESH_DAYS."""
        known = self.known.get(url)
        return known is not None and known.updated_at > self.cutoff

    def stale_links(self) -> list[PdfLink]:
        """Stored notices due for their weekly recompute."""
        links: list[PdfLink] = []
        start = 0
        while True:
            rows = (
                self.db.table("notices")
                .select("url,title,notice_date,source_page")
                .lt("updated_at", self.cutoff.isoformat())
                .order("id")
                .range(start, start + PAGE_SIZE - 1)
                .execute()
                .data
            )
            links += [PdfLink(r["url"], r["title"] or "", r["notice_date"], r["source_page"] or "") for r in rows]
            if len(rows) < PAGE_SIZE:
                return links
            start += PAGE_SIZE

    def record_failure(self, link: PdfLink, error: str) -> None:
        now = utcnow()
        row = (
            self.db.table("notices")
            .upsert(
                {
                    "url": link.url,
                    "title": link.title,
                    "notice_date": link.notice_date,
                    "source_page": link.source_page,
                    "status": "failed",
                    "error": error[:1000],
                    "updated_at": now.isoformat(),
                },
                on_conflict="url",
            )
            .execute()
            .data[0]
        )
        self.known[link.url] = KnownNotice(row["id"], "failed", None, now)

    def touch(self, url: str) -> None:
        """Content unchanged: just bump update_date on the notice and its vectors."""
        known = self.known[url]
        now = utcnow().isoformat()
        self.db.table("notices").update({"updated_at": now}).eq("id", known.id).execute()
        self.db.table("notice_chunks").update({"update_date": now}).eq("notice_id", known.id).execute()
        known.updated_at = utcnow()

    def save(
        self,
        link: PdfLink,
        status: str,
        num_pages: int,
        content_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
        error: str | None = None,
    ) -> None:
        now = utcnow()
        notice_id = (
            self.db.table("notices")
            .upsert(
                {
                    "url": link.url,
                    "title": link.title,
                    "notice_date": link.notice_date,
                    "source_page": link.source_page,
                    "status": status,
                    "error": error,
                    "content_hash": content_hash,
                    "num_pages": num_pages,
                    "updated_at": now.isoformat(),
                },
                on_conflict="url",
            )
            .execute()
            .data[0]["id"]
        )
        # Replace, don't append: the chunk count can change between versions of a PDF.
        self.db.table("notice_chunks").delete().eq("notice_id", notice_id).execute()
        rows = [
            {
                "notice_id": notice_id,
                "chunk_index": i,
                "content": chunk,
                "embedding": vector,
                "update_date": now.isoformat(),
            }
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
        ]
        for start in range(0, len(rows), INSERT_BATCH):
            self.db.table("notice_chunks").insert(rows[start : start + INSERT_BATCH]).execute()
        self.known[link.url] = KnownNotice(notice_id, status, content_hash, now)


def check_dim(vectors: list[list[float]]) -> list[list[float]]:
    if vectors and len(vectors[0]) != config.EMBED_DIM:
        raise ValueError(f"expected {config.EMBED_DIM}-dim embeddings, got {len(vectors[0])}")
    return vectors


class Embedder:
    """Local embeddings through Ollama (bge-m3 needs no task prefix)."""

    def __init__(self) -> None:
        self.host = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_EMBED_MODEL", config.EMBED_MODEL)
        self.client = ollama.Client(host=self.host)

    def check(self) -> None:
        try:
            self.client.show(self.model)
        except ollama.ResponseError as exc:
            raise SystemExit(f"Ollama model '{self.model}' not available ({exc}). Run: ollama pull {self.model}")
        except Exception as exc:
            raise SystemExit(f"Cannot reach Ollama at {self.host} ({exc}). Start it with: ollama serve")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
            batch = texts[start : start + config.EMBED_BATCH_SIZE]
            vectors.extend(self.client.embed(model=self.model, input=batch)["embeddings"])
        return check_dim(vectors)


class CloudflareEmbedder:
    """bge-m3 on Cloudflare Workers AI: identical vectors to local Ollama bge-m3 (cosine 1.0000
    in testing), but much faster than a CPU. Used when EMBED_PROVIDER=cloudflare."""

    def __init__(self) -> None:
        account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        if not account or not self.token:
            raise SystemExit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env to use EMBED_PROVIDER=cloudflare.")
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{config.CLOUDFLARE_EMBED_MODEL}"
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.token}"

    def check(self) -> None:
        try:
            self.embed_documents(["connection check"])
        except Exception as exc:
            raise SystemExit(f"Cloudflare Workers AI embedding failed: {exc}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), config.CLOUDFLARE_EMBED_BATCH_SIZE):
            batch = texts[start : start + config.CLOUDFLARE_EMBED_BATCH_SIZE]
            resp = self.session.post(self.url, json={"text": batch}, timeout=120)
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if not resp.ok or not body.get("success"):
                raise RuntimeError(f"Cloudflare returned {resp.status_code}: {body.get('errors') or resp.text[:200]}")
            vectors.extend(body["result"]["data"])
        return check_dim(vectors)


def make_embedder() -> Embedder | CloudflareEmbedder:
    provider = os.getenv("EMBED_PROVIDER", "ollama").strip().lower()
    return CloudflareEmbedder() if provider == "cloudflare" else Embedder()


def make_splitter() -> RecursiveCharacterTextSplitter:
    # Token-based: CHUNK_SIZE / CHUNK_OVERLAP are counted in tiktoken tokens, not characters.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


class Ingestor:
    def __init__(self, store: NoticeStore, embedder: Embedder, session: PoliteSession) -> None:
        self.store = store
        self.embedder = embedder
        self.session = session
        self.splitter = make_splitter()
        self.stats: Counter[str] = Counter()
        self.attempted: set[str] = set()

    def ingest(self, link: PdfLink) -> None:
        """Process one PDF. Never raises: every failure is logged and counted."""
        self.attempted.add(link.url)
        try:
            outcome = self._ingest(link)
        except Exception:
            log.exception("ERROR   unexpected failure for %s", link.url)
            outcome = "failed"
        self.stats[outcome] += 1

    def _ingest(self, link: PdfLink) -> str:
        existing = self.store.known.get(link.url)
        header = f"Notice: {link.title}" + (f" (dated {link.notice_date})" if link.notice_date else "")
        try:
            extracted = fetch_notice(self.session, link)
            status, num_pages, error = "ok", extracted.num_pages, None
            chunks = [f"{header}\n\n{chunk}" for chunk in self.splitter.split_text(extracted.text)]
        except NoTextError as exc:
            # Scanned PDF: index the title so the notice is still findable and linkable.
            log.warning("TITLE   %s <%s>: %s; storing title only", link.title, link.url, exc)
            status, num_pages, error = "title_only", exc.num_pages, str(exc)
            chunks = [f"{header}\n\n{TITLE_ONLY_NOTE}"]
        except ScrapeError as exc:
            log.error("FAILED  %s <%s>: %s", link.title, link.url, exc)
            # Keep a good copy if a refresh fails; it will be retried on the next run.
            if existing is None or existing.status == "failed":
                self.store.record_failure(link, str(exc))
            return "failed"

        content_hash = hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()
        if existing and existing.status == status and existing.content_hash == content_hash:
            self.store.touch(link.url)
            log.info("UNCHANGED %s", link.title)
            return "unchanged"

        embeddings = self.embedder.embed_documents(chunks)
        self.store.save(link, status, num_pages, content_hash, chunks, embeddings, error)
        if status == "title_only":
            return "title_only"
        log.info("STORED  %s (%d pages, %d chunks)", link.title, num_pages, len(chunks))
        return "updated" if existing else "stored"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape, embed and store university notices.")
    parser.add_argument("--url", action="append", help="Seed URL (repeatable). Defaults to config + urls.txt.")
    parser.add_argument("--max-pages", type=int, default=config.MAX_PAGES_PER_LISTING)
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Backfill: walk every page of every listing and don't stop at already-stored pages "
        "(stored PDFs are still skipped without downloading).",
    )
    parser.add_argument("--depth", type=int, default=config.CRAWL_MAX_DEPTH)
    parser.add_argument("--max-pdfs", type=int, default=0, help="Stop after this many PDFs (0 = no limit).")
    parser.add_argument("--no-refresh", action="store_true", help="Skip the weekly recompute of stale notices.")
    args = parser.parse_args()

    setup_logging()
    load_dotenv(config.BASE_DIR / ".env")
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY in .env (see .env.example).")
    url = normalize_supabase_url(url)

    store = NoticeStore(create_client(url, key))
    try:
        store.load_index()
    except Exception as exc:
        raise SystemExit(f"Could not read the notices table ({exc}). Did you run supabase/schema.sql?")
    embedder = make_embedder()
    embedder.check()

    session = PoliteSession()
    if args.all_pages:
        crawler = NoticeCrawler(session, max_pages=100_000, max_depth=args.depth, stop_when_known=False)
    else:
        crawler = NoticeCrawler(session, max_pages=args.max_pages, max_depth=args.depth)
    ingestor = Ingestor(store, embedder, session)
    seeds = args.url or config.get_notice_urls()
    log.info("Crawling %d seed page(s): %s", len(seeds), ", ".join(seeds))

    def budget_left() -> bool:
        return not args.max_pdfs or len(ingestor.attempted) < args.max_pdfs

    for link in crawler.crawl(seeds, should_skip=store.is_fresh):
        if not budget_left():
            log.info("Reached --max-pdfs %d", args.max_pdfs)
            break
        ingestor.ingest(link)

    if not args.no_refresh and budget_left():
        stale = [link for link in store.stale_links() if link.url not in ingestor.attempted]
        log.info("Weekly recompute: %d stored notice(s) older than %d days", len(stale), config.REFRESH_DAYS)
        for link in stale:
            if not budget_left():
                break
            ingestor.ingest(link)

    s = ingestor.stats
    log.info(
        "Done. listing pages=%d, new=%d, updated=%d, title only=%d, unchanged=%d, failed=%d, "
        "skipped (already stored)=%d",
        crawler.stats["pages"], s["stored"], s["updated"], s["title_only"], s["unchanged"], s["failed"],
        crawler.stats["skipped_known"],
    )


if __name__ == "__main__":
    main()
