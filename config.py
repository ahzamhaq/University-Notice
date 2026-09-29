"""Ingestion settings shared by scripts/scraper.py and scripts/embed_and_store.py."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# --- What to scrape ---------------------------------------------------------
# Seed pages that list notices. To add pages without touching code, put one URL
# per line in urls.txt (see EXTRA_URLS_FILE) or pass --url on the command line.
NOTICE_URLS = [
    "https://www.ipu.ac.in/notices.php",
    "https://www.ipu.ac.in/dsw_sports.php",
    # Add more notice pages here
]
EXTRA_URLS_FILE = BASE_DIR / "urls.txt"

# Only pages and PDFs on these hosts are fetched.
ALLOWED_DOMAINS = {"www.ipu.ac.in", "ipu.ac.in"}

# --- Crawling ---------------------------------------------------------------
# notices.php has 800+ pages, so pagination is capped. Newest notices come first.
MAX_PAGES_PER_LISTING = 5
# Stop paginating once a whole page contains only PDFs that are already stored.
STOP_WHEN_PAGE_ALREADY_KNOWN = True
# How many levels of linked sub-pages to follow from a seed (0 = seeds only).
# 0: only the pages listed in NOTICE_URLS / urls.txt are crawled (plus their own pagination).
CRAWL_MAX_DEPTH = 0
# Sub-page links are followed only when their URL matches one of these regexes.
FOLLOW_LINK_PATTERNS = [r"notice", r"circular", r"dsw_"]

REQUEST_DELAY = 2  # seconds between requests to the site
REQUEST_TIMEOUT = 60  # seconds
USER_AGENT = "UniversityNoticeRAG/1.0 (educational project)"

# --- PDFs -------------------------------------------------------------------
PDF_SIZE_LIMIT = 100 * 1024 * 1024  # 100MB
MIN_TEXT_CHARS = 50  # less than this after extraction = treat as scanned/empty

# --- Chunking & embeddings --------------------------------------------------
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens
# bge-m3 runs both locally (Ollama) and on Cloudflare Workers AI with identical vectors, so
# ingestion can stay local while the deployed site embeds questions on Cloudflare.
EMBED_MODEL = "bge-m3"  # Ollama model name; overridable with OLLAMA_EMBED_MODEL
CLOUDFLARE_EMBED_MODEL = "@cf/baai/bge-m3"
EMBED_DIM = 1024  # must match vector(1024) in supabase/schema.sql
EMBED_BATCH_SIZE = 32
CLOUDFLARE_EMBED_BATCH_SIZE = 50

# Stored notices older than this are re-downloaded and re-embedded (weekly recompute).
REFRESH_DAYS = 7


def get_notice_urls() -> list[str]:
    """NOTICE_URLS plus any non-comment lines from urls.txt, de-duplicated in order."""
    urls = list(NOTICE_URLS)
    if EXTRA_URLS_FILE.exists():
        for line in EXTRA_URLS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return list(dict.fromkeys(urls))
