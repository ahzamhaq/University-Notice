"""Live checks against the real Ollama, Supabase, ipu.ac.in and (optionally) the running API.

    pytest -m live                                  # everything live
    API_URL=http://localhost:3000 pytest -m live    # include the API end-to-end checks
Read-only: nothing is written to the database.
"""

import json
import os
from collections import Counter

import pytest
import requests
from dotenv import load_dotenv

import config
import scraper
from embed_and_store import normalize_supabase_url

pytestmark = pytest.mark.live
load_dotenv(config.BASE_DIR / ".env")
OLLAMA = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
API_URL = os.getenv("API_URL", "http://localhost:3000")


# --- fixtures ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def db():
    from supabase import create_client

    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url or not key:
        pytest.skip("SUPABASE_URL / SUPABASE_KEY not set")
    return create_client(normalize_supabase_url(url), key)


def fetch_all(db, table, columns):
    rows, start = [], 0
    while True:
        page = db.table(table).select(columns).order("id").range(start, start + 999).execute().data
        rows += page
        if len(page) < 1000:
            return rows
        start += 1000


@pytest.fixture(scope="session")
def notices(db):
    return fetch_all(db, "notices", "id,url,title,status,error,num_pages,updated_at")


@pytest.fixture(scope="session")
def chunk_counts(db):
    return Counter(r["notice_id"] for r in fetch_all(db, "notice_chunks", "id,notice_id"))


def embed_query(text):
    resp = requests.post(
        f"{OLLAMA}/api/embed", json={"model": "nomic-embed-text", "input": f"search_query: {text}"}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def search(db, question, count=5):
    """Same parameters as lib/rag.ts."""
    params = {
        "query_embedding": embed_query(question),
        "match_count": count,
        "match_threshold": 0.3,
        "max_per_notice": 1,
    }
    return db.rpc("match_notice_chunks", params).execute().data


@pytest.fixture(scope="session")
def api():
    try:
        requests.get(API_URL, timeout=5)
    except requests.RequestException:
        pytest.skip(f"API not running at {API_URL} (start it with npm run dev / next start)")
    return API_URL


# --- Ollama -------------------------------------------------------------------------------

def test_ollama_is_running():
    assert requests.get(f"{OLLAMA}/api/version", timeout=5).ok


def test_ollama_has_both_models():
    names = {m["name"].split(":")[0] for m in requests.get(f"{OLLAMA}/api/tags", timeout=5).json()["models"]}
    assert {"nomic-embed-text", os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")} <= names


def test_ollama_embedding_has_768_dims():
    assert len(embed_query("hello")) == config.EMBED_DIM


def test_ollama_chat_model_responds():
    resp = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": os.getenv("OLLAMA_CHAT_MODEL", "llama3.2"),
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "stream": False,
        },
        timeout=120,
    )
    assert resp.ok and resp.json()["message"]["content"].strip()


# --- Supabase data integrity ------------------------------------------------------------------

def test_database_has_notices_of_every_kind(notices):
    statuses = Counter(n["status"] for n in notices)
    assert statuses["ok"] > 0 and statuses["title_only"] > 0
    assert set(statuses) <= {"ok", "title_only", "failed"}


def test_no_duplicate_urls(notices):
    urls = [n["url"] for n in notices]
    assert len(urls) == len(set(urls))


def test_every_url_is_normalized(notices):
    assert all(scraper.normalize_url(n["url"]) == n["url"] and " " not in n["url"] for n in notices)


def test_every_indexed_notice_has_chunks(notices, chunk_counts):
    missing = [n["url"] for n in notices if n["status"] in ("ok", "title_only") and chunk_counts[n["id"]] == 0]
    assert missing == []


def test_title_only_notices_have_exactly_one_chunk(notices, chunk_counts):
    wrong = [n["url"] for n in notices if n["status"] == "title_only" and chunk_counts[n["id"]] != 1]
    assert wrong == []


def test_failed_notices_have_no_chunks_and_an_error(notices, chunk_counts):
    failed = [n for n in notices if n["status"] == "failed"]
    assert all(chunk_counts[n["id"]] == 0 and n["error"] for n in failed)


def test_no_orphan_chunks(notices, chunk_counts):
    assert set(chunk_counts) <= {n["id"] for n in notices}


def test_stored_embeddings_are_768_dims(db):
    row = db.table("notice_chunks").select("embedding,update_date,content").limit(1).execute().data[0]
    embedding = json.loads(row["embedding"]) if isinstance(row["embedding"], str) else row["embedding"]
    assert len(embedding) == config.EMBED_DIM
    assert row["update_date"] and row["content"].startswith("Notice: ")


def test_all_chunks_have_update_date(db):
    missing = db.table("notice_chunks").select("id", count="exact").is_("update_date", "null").limit(1).execute()
    assert missing.count == 0


def test_rls_blocks_anonymous_reads():
    """The anon key (if configured) must not be able to read the tables."""
    anon = os.getenv("SUPABASE_ANON_KEY")
    if not anon:
        pytest.skip("SUPABASE_ANON_KEY not set; RLS check skipped")
    url = normalize_supabase_url(os.getenv("SUPABASE_URL", ""))
    rows = requests.get(f"{url}/rest/v1/notices?select=id&limit=1", headers={"apikey": anon}, timeout=10).json()
    assert rows == []


# --- retrieval -------------------------------------------------------------------------------

def test_search_results_are_sorted_by_score_and_above_threshold(db):
    results = search(db, "examination datesheet")
    scores = [r["score"] for r in results]
    assert results and scores == sorted(scores, reverse=True)
    assert min(r["similarity"] for r in results) > 0.3
    assert all(r["score"] >= r["similarity"] for r in results)  # boost is never negative


def test_recency_boost_prefers_new_notices_over_near_identical_old_ones(db):
    """The April 2025 datesheet used to outrank four Sep 2026 ones by 0.001 similarity."""
    dates = [r["notice_date"] for r in search(db, "latest exam datesheet")]
    assert all(d and d >= "2026-09-01" for d in dates), dates


def test_clearly_better_old_match_still_wins(db):
    """The only Yogasana championship notice is from Oct 2025; newer sports notices mustn't bury it."""
    top = search(db, "Inter-collegiate yogasana championship")[0]
    assert "yogasana" in top["title"].lower()


@pytest.mark.parametrize("question", ["selection trials", "scholarship", "EWS financial assistance interview"])
def test_no_single_notice_floods_the_results(db, question):
    per_notice = Counter(r["notice_id"] for r in search(db, question))
    assert per_notice and max(per_notice.values()) == 1


@pytest.mark.parametrize(
    "question, expected_in_title",
    [
        ("Selection trials for the women's football team", "Women's Football"),
        ("How do I apply for the NSP scholarship?", "Scholarship"),
        ("Financial assistance for economically weaker section students", "EWS"),
        ("MCA refund order for withdrawal candidates", "MCA"),
        ("Professional development visit for chemistry teachers", "Professional Development Visit"),
        ("Inter-collegiate yogasana championship", "Yogasana"),
        ("LLB exam datesheet December 2026", "LLB"),
    ],
)
def test_retrieval_finds_the_right_notice(db, question, expected_in_title):
    titles = [r["title"] for r in search(db, question)]
    assert any(expected_in_title.lower() in (t or "").lower() for t in titles), titles


# --- live site ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def site():
    return scraper.PoliteSession()


@pytest.mark.parametrize("page", ["notices.php", "dsw_sports.php"])
def test_live_listing_still_parses(site, page):
    url = f"https://www.ipu.ac.in/{page}"
    pdfs, next_url, _ = scraper.parse_listing(site.get(url).content, url)
    assert len(pdfs) >= 20, "site layout may have changed"
    assert next_url and "page=2" in next_url
    assert sum(p.notice_date is not None for p in pdfs) >= 20


def test_robots_allows_our_crawler(site):
    assert site._allowed("https://www.ipu.ac.in/notices.php")


def test_live_pdf_download_and_extract(site, notices):
    ok = next(n for n in notices if n["status"] == "ok" and (n["num_pages"] or 0) <= 3)
    extracted = scraper.fetch_notice(site, scraper.PdfLink(ok["url"], ok["title"], None, ""))
    assert len(extracted.text) >= config.MIN_TEXT_CHARS


# --- API end-to-end -----------------------------------------------------------------------

def test_homepage_renders(api):
    resp = requests.get(api, timeout=30)
    assert resp.ok and "IPU Notice Search" in resp.text


def test_api_rejects_bad_input(api):
    resp = requests.post(f"{api}/api/query", json={"question": "a"}, timeout=30)
    assert resp.status_code == 400 and "error" in resp.json()


@pytest.mark.parametrize(
    "question, expected_in_title",
    [
        ("Is there a datesheet for LLB exams?", "LLB"),
        ("When are the selection trials for the women's volleyball team?", "Volleyball"),
        # Regression: an older, scanned (title-only) notice that is the clear best match must be [1].
        ("Are there any kabaddi selection trials for women?", "Kabaddi"),
    ],
)
def test_api_answers_with_relevant_sources(api, question, expected_in_title):
    resp = requests.post(f"{api}/api/query", json={"question": question}, timeout=180)
    assert resp.ok, resp.text
    body = resp.json()
    assert len(body["answer"]) > 20
    assert expected_in_title.lower() in body["sources"][0]["title"].lower(), body["sources"]
    assert all(s["url"].startswith("https://www.ipu.ac.in/") for s in body["sources"])
    assert resp.headers["X-RateLimit-Limit"] == "50"


def test_api_stream_sends_sources_fast_then_tokens(api):
    """Sources must arrive in seconds (retrieval only), before the slow model starts writing."""
    import time

    started = time.monotonic()
    first_event_at, events = None, []
    with requests.post(
        f"{api}/api/query",
        json={"question": "latest exam datesheet"},
        headers={"Accept": "application/x-ndjson"},
        stream=True,
        timeout=400,
    ) as resp:
        assert resp.ok and "application/x-ndjson" in resp.headers["Content-Type"]
        for line in resp.iter_lines():
            if line:
                events.append(json.loads(line))
                first_event_at = first_event_at or time.monotonic() - started

    types = [e["type"] for e in events]
    assert types[0] == "sources" and types[-1] == "done" and "token" in types, types
    assert first_event_at < 20, f"sources took {first_event_at:.1f}s"
    assert all((s["noticeDate"] or "") >= "2026-09-01" for s in events[0]["sources"])  # recency boost
    assert len("".join(e["text"] for e in events if e["type"] == "token")) > 20

