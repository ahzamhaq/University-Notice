"""Crawl IPU notice pages, find PDF links, download them and extract their text.

Used as a library by embed_and_store.py. It can also run on its own as a dry run
that writes the extracted text to data/extracted.jsonl instead of Supabase:

    python scripts/scraper.py --max-pages 1 --max-pdfs 5
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from PyPDF2 import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("scraper")

DATE_RE = re.compile(r"\b(\d{1,2})[-./](\d{1,2})[-./](\d{4})\b")
NEXT_RE = re.compile(r"\bnext\b|»", re.IGNORECASE)


class ScrapeError(Exception):
    """A page or PDF that could not be fetched or parsed. Logged and recorded, never fatal."""


class NoTextError(ScrapeError):
    """The PDF opened fine but has no text layer (usually a scanned image)."""

    def __init__(self, message: str, num_pages: int) -> None:
        super().__init__(message)
        self.num_pages = num_pages


@dataclass
class PdfLink:
    url: str
    title: str
    notice_date: str | None  # ISO date (YYYY-MM-DD) from the listing row, if present
    source_page: str


@dataclass
class ExtractedPdf:
    text: str
    num_pages: int


def setup_logging() -> None:
    """Log to the console and to logs/ingest.log (rotating)."""
    root = logging.getLogger()
    if root.handlers:
        return
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Notice titles contain characters the Windows console codepage can't encode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = RotatingFileHandler(
        config.LOG_DIR / "ingest.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    logfile.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(logfile)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "urllib3", "PyPDF2"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication: no fragment, unsafe characters percent-encoded."""
    return requests.utils.requote_uri(urldefrag(url.strip()).url)


def in_scope(url: str) -> bool:
    parts = urlparse(url)
    return parts.scheme in ("http", "https") and parts.netloc.lower() in config.ALLOWED_DOMAINS


def is_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def parse_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    day, month, year = map(int, match.groups())
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None


class PoliteSession:
    """requests.Session that honours robots.txt and waits REQUEST_DELAY between requests."""

    def __init__(self, delay: float = config.REQUEST_DELAY) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.delay = delay
        self._last_request = 0.0
        self._robots: dict[str, RobotFileParser | None] = {}

    def get(self, url: str, **kwargs) -> requests.Response:
        if not self._allowed(url):
            raise ScrapeError(f"disallowed by robots.txt: {url}")
        return self._request(url, **kwargs)

    def _request(self, url: str, **kwargs) -> requests.Response:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.session.get(url, timeout=config.REQUEST_TIMEOUT, **kwargs)
        finally:
            self._last_request = time.monotonic()

    def _allowed(self, url: str) -> bool:
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(config.USER_AGENT, url)

    def _load_robots(self, origin: str) -> RobotFileParser | None:
        """Parse robots.txt; a missing or unreachable file means everything is allowed."""
        try:
            resp = self._request(f"{origin}/robots.txt")
        except requests.RequestException as exc:
            log.warning("Could not fetch robots.txt for %s (%s); assuming allowed", origin, exc)
            return None
        if resp.status_code >= 400:
            return None
        parser = RobotFileParser()
        parser.parse(resp.text.splitlines())
        crawl_delay = parser.crawl_delay(config.USER_AGENT)
        if crawl_delay and float(crawl_delay) > self.delay:
            log.info("robots.txt asks for a %ss crawl delay on %s", crawl_delay, origin)
            self.delay = float(crawl_delay)
        return parser


def _is_navigation(anchor: Tag) -> bool:
    """Links in site menus (not the notice table) are ignored."""
    if anchor.find_parent(["nav", "header", "footer"]) is not None:
        return True
    return any("dropdown" in cls or "nav-link" in cls for cls in anchor.get("class", []))


def parse_listing(html: bytes, page_url: str) -> tuple[list[PdfLink], str | None, list[str]]:
    """Return (PDF links, next-page URL, sub-page URLs worth following) for a listing page."""
    soup = BeautifulSoup(html, "html.parser")
    pdfs: list[PdfLink] = []
    next_url: str | None = None
    subpages: list[str] = []
    follow = [re.compile(p, re.IGNORECASE) for p in config.FOLLOW_LINK_PATTERNS]
    own_path = urlparse(page_url).path

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        url = normalize_url(urljoin(page_url, href))
        if not in_scope(url) or _is_navigation(anchor):
            continue
        text = " ".join(anchor.get_text(" ", strip=True).split())

        if is_pdf(url):
            row = anchor.find_parent("tr")
            date = parse_date(row.get_text(" ", strip=True)) if row else None
            title = text or Path(urlparse(url).path).stem
            pdfs.append(PdfLink(url=url, title=title, notice_date=date, source_page=page_url))
        elif next_url is None and NEXT_RE.search(text):
            next_url = url
        elif urlparse(url).path != own_path and any(p.search(url) for p in follow):
            # Same path with another query is this listing's pagination, not a sub-page.
            subpages.append(url)

    return pdfs, next_url, list(dict.fromkeys(subpages))


class NoticeCrawler:
    """Walks seed listing pages (with pagination) and linked sub-pages, yielding new PDF links."""

    def __init__(
        self,
        session: PoliteSession,
        max_pages: int = config.MAX_PAGES_PER_LISTING,
        max_depth: int = config.CRAWL_MAX_DEPTH,
        stop_when_known: bool = config.STOP_WHEN_PAGE_ALREADY_KNOWN,
    ) -> None:
        self.session = session
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.stop_when_known = stop_when_known
        self.stats: Counter[str] = Counter()

    def crawl(
        self, seeds: Iterable[str], should_skip: Callable[[str], bool] = lambda url: False
    ) -> Iterator[PdfLink]:
        """Yield each PDF link once. `should_skip(url)` lets the caller drop already-stored URLs
        before anything is downloaded."""
        visited: set[str] = set()
        seen_pdfs: set[str] = set()
        queue = deque((normalize_url(seed), 0) for seed in seeds)

        while queue:
            page_url, depth = queue.popleft()
            pages = 0
            while page_url and pages < self.max_pages and page_url not in visited:
                visited.add(page_url)
                pages += 1
                try:
                    resp = self.session.get(page_url)
                    resp.raise_for_status()
                except (requests.RequestException, ScrapeError) as exc:
                    log.error("Failed to fetch listing %s: %s", page_url, exc)
                    self.stats["pages_failed"] += 1
                    break
                self.stats["pages"] += 1

                pdfs, next_url, subpages = parse_listing(resp.content, page_url)
                log.info("Listing %s (depth %d, page %d): %d PDF links", page_url, depth, pages, len(pdfs))

                new_on_page = 0
                for link in pdfs:
                    if link.url in seen_pdfs:
                        continue
                    seen_pdfs.add(link.url)
                    if should_skip(link.url):
                        self.stats["skipped_known"] += 1
                        continue
                    new_on_page += 1
                    yield link

                if depth < self.max_depth:
                    queue.extend((sub, depth + 1) for sub in subpages if sub not in visited)

                if self.stop_when_known and pdfs and new_on_page == 0:
                    log.info("Every PDF on %s is already stored; stopping pagination here", page_url)
                    break
                page_url = next_url


def download_pdf(session: PoliteSession, url: str) -> bytes:
    """Download a PDF into memory, refusing anything over PDF_SIZE_LIMIT."""
    limit = config.PDF_SIZE_LIMIT
    limit_mb = limit // (1024 * 1024)
    buf = io.BytesIO()
    try:
        with session.get(url, stream=True) as resp:
            resp.raise_for_status()
            try:
                declared = int(resp.headers.get("Content-Length", 0))
            except ValueError:
                declared = 0
            if declared > limit:
                raise ScrapeError(f"PDF is {declared / 1024 / 1024:.1f}MB, over the {limit_mb}MB limit")
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                buf.write(chunk)
                if buf.tell() > limit:
                    raise ScrapeError(f"PDF exceeded the {limit_mb}MB limit while downloading")
    except requests.RequestException as exc:
        raise ScrapeError(f"download failed: {exc}") from exc

    data = buf.getvalue()
    if b"%PDF" not in data[:1024]:
        raise ScrapeError("response is not a PDF")
    return data


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")  # Postgres rejects NUL characters
    lines = (" ".join(line.split()) for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_text(data: bytes) -> ExtractedPdf:
    """Extract text from PDF bytes. Raises ScrapeError for unreadable or text-less PDFs."""
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ScrapeError("PDF is password protected")
        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # one bad page shouldn't lose the rest
                log.warning("Could not extract page %d: %s", number, exc)
        num_pages = len(reader.pages)
    except ScrapeError:
        raise
    except Exception as exc:
        raise ScrapeError(f"could not parse PDF: {exc}") from exc

    text = clean_text("\n".join(pages))
    if len(text) < config.MIN_TEXT_CHARS:
        raise NoTextError("no extractable text (probably a scanned image)", num_pages)
    return ExtractedPdf(text=text, num_pages=num_pages)


def fetch_notice(session: PoliteSession, link: PdfLink) -> ExtractedPdf:
    return extract_text(download_pdf(session, link.url))


# --- Standalone dry run -------------------------------------------------------

def _load_done_urls(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape notice PDFs to data/extracted.jsonl (no database).")
    parser.add_argument("--url", action="append", help="Seed URL (repeatable). Defaults to config + urls.txt.")
    parser.add_argument("--max-pages", type=int, default=config.MAX_PAGES_PER_LISTING)
    parser.add_argument("--depth", type=int, default=config.CRAWL_MAX_DEPTH)
    parser.add_argument("--max-pdfs", type=int, default=0, help="Stop after this many PDFs (0 = no limit).")
    parser.add_argument("--output", type=Path, default=config.DATA_DIR / "extracted.jsonl")
    args = parser.parse_args()

    setup_logging()
    seeds = args.url or config.get_notice_urls()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_urls(args.output)
    log.info("Dry run: %d seed(s), %d PDFs already in %s", len(seeds), len(done), args.output)

    crawler = NoticeCrawler(PoliteSession(), max_pages=args.max_pages, max_depth=args.depth)
    stats: Counter[str] = Counter()
    with args.output.open("a", encoding="utf-8") as out:
        for link in crawler.crawl(seeds, should_skip=done.__contains__):
            if args.max_pdfs and stats["attempted"] >= args.max_pdfs:
                break
            stats["attempted"] += 1
            record = asdict(link) | {"scraped_at": datetime.now(timezone.utc).isoformat()}
            try:
                extracted = fetch_notice(crawler.session, link)
                record |= {"status": "ok", "num_pages": extracted.num_pages, "text": extracted.text}
                stats["ok"] += 1
                log.info("OK      %s (%d pages, %d chars)", link.title, extracted.num_pages, len(extracted.text))
            except NoTextError as exc:
                record |= {"status": "title_only", "num_pages": exc.num_pages, "error": str(exc)}
                stats["title_only"] += 1
                log.warning("TITLE   %s <%s>: %s; keeping title only", link.title, link.url, exc)
            except ScrapeError as exc:
                record |= {"status": "failed", "error": str(exc)}
                stats["failed"] += 1
                log.error("FAILED  %s <%s>: %s", link.title, link.url, exc)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(
        "Done. pages=%d ok=%d title_only=%d failed=%d skipped_known=%d",
        crawler.stats["pages"], stats["ok"], stats["title_only"], stats["failed"],
        crawler.stats["skipped_known"],
    )


if __name__ == "__main__":
    main()
