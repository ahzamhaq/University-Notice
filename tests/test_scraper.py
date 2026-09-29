"""Offline unit tests for scripts/scraper.py and config.py."""

import re
import time

import pytest
import requests

import config
import scraper
from conftest import FakeResponse, FakeSession
from scraper import NoTextError, NoticeCrawler, ScrapeError

BASE = "https://www.ipu.ac.in"

LISTING_HTML = """
<html><body>
<nav><a href="/nav-notice.pdf">Menu PDF in nav</a></nav>
<div class="dropdown-menu"><a class="dropdown-item" href="/Pubinfo2025/refund2025.pdf">Refund (menu)</a></div>
<a class="dropdown-item" href="/notices_menu.php">Notices menu page</a>
<table>
  <tr><th>Title/Notices</th><th>Uploading Date</th></tr>
  <tbody>
    <tr><td><a href="/Pubinfo2026/nt280925545 (6).pdf">Professional Development Visit, USBAS</a></td><td>28-09-2026</td></tr>
    <tr><td><a href="/Pubinfo2026/nt280925545 (5).pdf">Schedule of 2nd Counselling</a></td><td>27-09-2026</td></tr>
    <tr><td><a href="/Pubinfo2026/nt280925545 (5).pdf">Schedule of 2nd Counselling</a></td><td>27-09-2026</td></tr>
    <tr><td><a href="/Pubinfo2026/untitled.pdf"></a></td><td>no date here</td></tr>
  </tbody>
</table>
<a href="https://example.com/other.pdf">External PDF</a>
<a href="/dsw_student.php">Student welfare</a>
<a href="/aboutuni.php">About</a>
<a href="?page=1&limit=25">&laquo; Previous</a>
<a href="?page=2&limit=25">Next &raquo;</a>
<a href="mailto:x@ipu.ac.in">Mail</a>
<footer><a href="/footer.pdf">Footer PDF</a></footer>
</body></html>
"""


@pytest.fixture
def parsed():
    return scraper.parse_listing(LISTING_HTML.encode(), f"{BASE}/notices.php")


# --- dates --------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("28-09-2026", "2026-09-28"),
        ("Uploaded 5/1/2027 by admin", "2027-01-05"),
        ("12.03.2025", "2025-03-12"),
        ("31-02-2026", None),  # impossible date
        ("no date here", None),
    ],
)
def test_parse_date(text, expected):
    assert scraper.parse_date(text) == expected


# --- URL helpers ----------------------------------------------------------------

def test_normalize_url_encodes_spaces():
    assert scraper.normalize_url(f"{BASE}/Pubinfo2026/nt1 (6).pdf") == f"{BASE}/Pubinfo2026/nt1%20(6).pdf"


def test_normalize_url_is_idempotent():
    once = scraper.normalize_url(f"{BASE}/a b.pdf")
    assert scraper.normalize_url(once) == once  # no double-encoding -> dedup stays stable


def test_normalize_url_strips_fragment_and_whitespace():
    assert scraper.normalize_url(f"  {BASE}/notices.php#top ") == f"{BASE}/notices.php"


@pytest.mark.parametrize(
    "url, expected",
    [
        (f"{BASE}/notices.php", True),
        ("https://ipu.ac.in/x.pdf", True),
        ("https://example.com/x.pdf", False),
        ("ftp://www.ipu.ac.in/x.pdf", False),
    ],
)
def test_in_scope(url, expected):
    assert scraper.in_scope(url) is expected


def test_is_pdf_is_case_insensitive_and_ignores_query():
    assert scraper.is_pdf(f"{BASE}/A.PDF?v=2")
    assert not scraper.is_pdf(f"{BASE}/notices.php?file=a.pdf")


# --- listing parser -----------------------------------------------------------------

def test_listing_extracts_table_pdfs_with_title_and_date(parsed):
    pdfs, _, _ = parsed
    first = pdfs[0]
    assert first.url == f"{BASE}/Pubinfo2026/nt280925545%20(6).pdf"
    assert first.title == "Professional Development Visit, USBAS"
    assert first.notice_date == "2026-09-28"
    assert first.source_page == f"{BASE}/notices.php"


def test_listing_skips_menu_nav_footer_and_external_pdfs(parsed):
    urls = {p.url for p in parsed[0]}
    assert not any(bad in u for u in urls for bad in ("refund2025", "nav-notice", "footer", "example.com"))
    assert len(urls) == 3  # the three table PDFs (one appears twice in the table)


def test_listing_untitled_pdf_falls_back_to_filename(parsed):
    untitled = [p for p in parsed[0] if p.url.endswith("untitled.pdf")][0]
    assert untitled.title == "untitled"
    assert untitled.notice_date is None


def test_listing_finds_next_link_not_previous(parsed):
    assert parsed[1] == f"{BASE}/notices.php?page=2&limit=25"


def test_listing_subpages_follow_patterns_only(parsed):
    subpages = parsed[2]
    assert f"{BASE}/dsw_student.php" in subpages
    assert f"{BASE}/aboutuni.php" not in subpages  # doesn't match FOLLOW_LINK_PATTERNS
    assert f"{BASE}/notices_menu.php" not in subpages  # menu link


def test_listing_pagination_is_not_a_subpage(parsed):
    assert not any("page=" in s for s in parsed[2])


# --- text extraction ----------------------------------------------------------------

def test_clean_text_removes_nul_and_collapses_whitespace():
    assert scraper.clean_text("a\x00b   c\n\n\n\n\nd  ") == "ab c\n\nd"


def test_extract_text_from_real_text_pdf(text_pdf):
    result = scraper.extract_text(text_pdf)
    assert "LLB examinations" in result.text
    assert result.num_pages == 1


def test_scanned_pdf_raises_no_text_error_with_page_count(blank_pdf):
    with pytest.raises(NoTextError) as exc:
        scraper.extract_text(blank_pdf)
    assert exc.value.num_pages == 2


def test_corrupt_pdf_raises_scrape_error_not_no_text():
    with pytest.raises(ScrapeError) as exc:
        scraper.extract_text(b"%PDF-1.4 this is garbage")
    assert not isinstance(exc.value, NoTextError)


def test_no_text_error_is_a_scrape_error():
    assert issubclass(NoTextError, ScrapeError)


# --- downloads ------------------------------------------------------------------------

URL = f"{BASE}/x.pdf"


def test_download_returns_pdf_bytes(text_pdf):
    assert scraper.download_pdf(FakeSession({URL: FakeResponse(text_pdf)}), URL) == text_pdf


def test_download_rejects_declared_size_over_limit():
    huge = FakeResponse(b"%PDF", headers={"Content-Length": str(101 * 1024 * 1024)})
    with pytest.raises(ScrapeError, match="over the 100MB limit"):
        scraper.download_pdf(FakeSession({URL: huge}), URL)


def test_download_enforces_limit_while_streaming(monkeypatch):
    monkeypatch.setattr(config, "PDF_SIZE_LIMIT", 100 * 1024)
    body = b"%PDF" + b"x" * (200 * 1024)  # no Content-Length header
    with pytest.raises(ScrapeError, match="while downloading"):
        scraper.download_pdf(FakeSession({URL: FakeResponse(body)}), URL)


def test_download_rejects_html_masquerading_as_pdf():
    with pytest.raises(ScrapeError, match="not a PDF"):
        scraper.download_pdf(FakeSession({URL: FakeResponse("<html>Error</html>")}), URL)


def test_download_404_becomes_scrape_error():
    with pytest.raises(ScrapeError, match="download failed"):
        scraper.download_pdf(FakeSession({}), URL)


# --- PoliteSession ------------------------------------------------------------------------

def _polite(monkeypatch, robots: FakeResponse, delay: float = 0.0):
    session = scraper.PoliteSession(delay=delay)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, time.monotonic()))
        return robots if url.endswith("/robots.txt") else FakeResponse("ok")

    monkeypatch.setattr(session.session, "get", fake_get)
    return session, calls


def test_polite_session_waits_between_requests(monkeypatch):
    session, calls = _polite(monkeypatch, FakeResponse(status=404), delay=0.2)
    session.get(f"{BASE}/a")
    session.get(f"{BASE}/b")
    gaps = [b[1] - a[1] for a, b in zip(calls, calls[1:])]
    assert len(calls) == 3  # robots.txt + 2 pages
    assert all(gap >= 0.19 for gap in gaps)


def test_polite_session_honours_robots_disallow(monkeypatch):
    session, _ = _polite(monkeypatch, FakeResponse("User-agent: *\nDisallow: /private/"))
    with pytest.raises(ScrapeError, match="robots.txt"):
        session.get(f"{BASE}/private/x.pdf")
    assert session.get(f"{BASE}/public.pdf").text == "ok"


def test_polite_session_missing_robots_allows_everything(monkeypatch):
    session, calls = _polite(monkeypatch, FakeResponse(status=404))
    session.get(f"{BASE}/anything")
    session.get(f"{BASE}/else")
    assert sum(url.endswith("robots.txt") for url, _ in calls) == 1  # fetched once, cached


def test_polite_session_adopts_larger_crawl_delay(monkeypatch):
    session, _ = _polite(monkeypatch, FakeResponse("User-agent: *\nCrawl-delay: 5"), delay=0)
    session._allowed(f"{BASE}/x")
    assert session.delay == 5


# --- crawler ---------------------------------------------------------------------------------

def listing(pdfs: list[str], next_href: str | None = None, links: list[str] = ()) -> FakeResponse:
    rows = "".join(f'<tr><td><a href="{p}">{p}</a></td><td>01-01-2026</td></tr>' for p in pdfs)
    extra = "".join(f'<a href="{href}">link</a>' for href in links)
    nxt = f'<a href="{next_href}">Next &raquo;</a>' if next_href else ""
    return FakeResponse(f"<table>{rows}</table>{nxt}{extra}")


def crawl(pages, seeds, should_skip=lambda url: False, **kwargs):
    kwargs.setdefault("max_pages", 5)
    kwargs.setdefault("max_depth", 0)
    crawler = NoticeCrawler(FakeSession(pages), **kwargs)
    return crawler, [link.url.rsplit("/", 1)[-1] for link in crawler.crawl(seeds, should_skip=should_skip)]


def test_crawler_follows_pagination_up_to_max_pages():
    pages = {f"{BASE}/n.php?page={i}": listing([f"/p{i}.pdf"], f"?page={i + 1}") for i in range(1, 10)}
    crawler, got = crawl(pages, [f"{BASE}/n.php?page=1"], max_pages=3)
    assert got == ["p1.pdf", "p2.pdf", "p3.pdf"]
    assert crawler.stats["pages"] == 3


def test_crawler_stops_when_a_page_is_entirely_known():
    pages = {
        f"{BASE}/n.php": listing(["/new.pdf"], "?page=2"),
        f"{BASE}/n.php?page=2": listing(["/old1.pdf", "/old2.pdf"], "?page=3"),
        f"{BASE}/n.php?page=3": listing(["/older.pdf"]),
    }
    crawler, got = crawl(pages, [f"{BASE}/n.php"], should_skip=lambda u: "old" in u)
    assert got == ["new.pdf"]
    assert crawler.stats["pages"] == 2 and crawler.stats["skipped_known"] == 2


def test_crawler_yields_each_pdf_once_across_pages():
    pages = {
        f"{BASE}/n.php": listing(["/a.pdf", "/a.pdf", "/b.pdf"], "?page=2"),
        f"{BASE}/n.php?page=2": listing(["/b.pdf", "/c.pdf"]),
    }
    _, got = crawl(pages, [f"{BASE}/n.php"])
    assert got == ["a.pdf", "b.pdf", "c.pdf"]


def test_backfill_mode_continues_past_fully_known_pages():
    pages = {
        f"{BASE}/n.php": listing(["/old1.pdf"], "?page=2"),
        f"{BASE}/n.php?page=2": listing(["/old2.pdf"], "?page=3"),
        f"{BASE}/n.php?page=3": listing(["/missing.pdf"]),
    }
    crawler, got = crawl(pages, [f"{BASE}/n.php"], should_skip=lambda u: "old" in u, stop_when_known=False)
    assert got == ["missing.pdf"] and crawler.stats["pages"] == 3


def test_depth_zero_visits_only_seed_pages_and_their_pagination():
    pages = {
        f"{BASE}/notices.php": listing(["/a.pdf"], "?page=2", links=["/notice_sub.php"]),
        f"{BASE}/notices.php?page=2": listing(["/b.pdf"]),
        f"{BASE}/notice_sub.php": listing(["/should-not-appear.pdf"]),
    }
    crawler, got = crawl(pages, [f"{BASE}/notices.php"], max_depth=0)
    assert got == ["a.pdf", "b.pdf"]
    assert f"{BASE}/notice_sub.php" not in crawler.session.requested


def test_configured_seeds_are_exactly_the_requested_pages():
    urls = config.get_notice_urls()
    assert config.CRAWL_MAX_DEPTH == 0
    assert len(urls) == 27  # notices.php + the 26 pages in urls.txt (dsw_sports.php listed in both)
    assert all(u.startswith("https://www.ipu.ac.in/") for u in urls)


def test_crawler_recurses_to_depth_limit_only():
    pages = {
        f"{BASE}/notices.php": listing(["/root.pdf"], links=["/notice_sub.php"]),
        f"{BASE}/notice_sub.php": listing(["/sub.pdf"], links=["/notice_deeper.php"]),
        f"{BASE}/notice_deeper.php": listing(["/deep.pdf"]),
    }
    _, got = crawl(pages, [f"{BASE}/notices.php"], max_depth=1)
    assert got == ["root.pdf", "sub.pdf"]


def test_crawler_survives_failed_listing_and_continues_with_next_seed():
    pages = {
        f"{BASE}/broken.php": requests.ConnectionError("boom"),
        f"{BASE}/ok.php": listing(["/fine.pdf"]),
    }
    crawler, got = crawl(pages, [f"{BASE}/broken.php", f"{BASE}/ok.php"])
    assert got == ["fine.pdf"]
    assert crawler.stats["pages_failed"] == 1


# --- config -----------------------------------------------------------------------------------

def test_get_notice_urls_merges_urls_txt(monkeypatch, tmp_path):
    extra = tmp_path / "urls.txt"
    extra.write_text(
        "# comment\n\nhttps://www.ipu.ac.in/extra.php\nhttps://www.ipu.ac.in/notices.php\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "EXTRA_URLS_FILE", extra)
    urls = config.get_notice_urls()
    assert urls[: len(config.NOTICE_URLS)] == config.NOTICE_URLS
    assert urls.count("https://www.ipu.ac.in/notices.php") == 1
    assert "https://www.ipu.ac.in/extra.php" in urls and not any(u.startswith("#") for u in urls)


def test_config_matches_requirements():
    assert (config.CHUNK_SIZE, config.CHUNK_OVERLAP) == (500, 50)
    assert config.PDF_SIZE_LIMIT == 100 * 1024 * 1024
    assert config.REQUEST_DELAY >= 2
    assert config.REFRESH_DAYS == 7


def test_schema_vector_size_and_statuses_match_code():
    schema = (config.BASE_DIR / "supabase" / "schema.sql").read_text(encoding="utf-8")
    assert set(re.findall(r"vector\((\d+)\)", schema)) == {str(config.EMBED_DIM)}
    assert "'ok', 'title_only', 'failed'" in schema
