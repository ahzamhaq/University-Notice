"""Offline unit tests for scripts/embed_and_store.py (fake store, fake embedder)."""

import json as json_module
from datetime import timedelta

import pytest
import tiktoken

import config
import embed_and_store as ingest
from conftest import FakeResponse
from embed_and_store import Embedder, Ingestor, KnownNotice, NoticeStore, utcnow
from scraper import ExtractedPdf, NoTextError, PdfLink, ScrapeError

LINK = PdfLink("https://www.ipu.ac.in/a.pdf", "LLB Datesheet", "2026-09-28", "https://www.ipu.ac.in/notices.php")


class FakeStore:
    def __init__(self, known=None):
        self.known = known or {}
        self.saved, self.failures, self.touched = [], [], []

    def save(self, link, status, num_pages, content_hash, chunks, embeddings, error=None):
        self.saved.append(dict(status=status, num_pages=num_pages, hash=content_hash, chunks=chunks, error=error))
        self.known[link.url] = KnownNotice(1, status, content_hash, utcnow())

    def record_failure(self, link, error):
        self.failures.append(error)

    def touch(self, url):
        self.touched.append(url)


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[0.0] * config.EMBED_DIM for _ in texts]


def make_ingestor(monkeypatch, fetch, store=None):
    monkeypatch.setattr(ingest, "fetch_notice", fetch)
    embedder = FakeEmbedder()
    return Ingestor(store or FakeStore(), embedder, session=None), embedder


def returns(text, pages=1):
    return lambda session, link: ExtractedPdf(text, pages)


def raises(exc):
    def fetch(session, link):
        raise exc

    return fetch


# --- helpers ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "https://abc.supabase.co",
        "https://abc.supabase.co/",
        "https://abc.supabase.co/rest/v1/",
        "  https://abc.supabase.co/rest/v1  ",
    ],
)
def test_normalize_supabase_url(raw):
    assert ingest.normalize_supabase_url(raw) == "https://abc.supabase.co"


@pytest.mark.parametrize(
    "value",
    ["2026-09-29T10:00:00+00:00", "2026-09-29T10:00:00.12345+00:00", "2026-09-29T10:00:00.1Z"],
)
def test_parse_ts_handles_postgres_formats(value):
    parsed = ingest.parse_ts(value)
    assert (parsed.year, parsed.hour, parsed.utcoffset()) == (2026, 10, timedelta(0))


def test_chunks_respect_500_token_limit_with_overlap():
    words = " ".join(f"word{i}" for i in range(3000))
    chunks = ingest.make_splitter().split_text(words)
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(chunks) > 1
    assert all(len(enc.encode(c)) <= config.CHUNK_SIZE for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # consecutive chunks overlap


def test_store_is_fresh_logic():
    store = NoticeStore(client=None)
    store.known = {
        "fresh": KnownNotice(1, "ok", "h", utcnow() - timedelta(days=1)),
        "stale": KnownNotice(2, "ok", "h", utcnow() - timedelta(days=8)),
    }
    assert store.is_fresh("fresh")
    assert not store.is_fresh("stale")
    assert not store.is_fresh("unknown")


# --- ingest outcomes --------------------------------------------------------------------

def test_text_pdf_is_chunked_with_title_header_and_saved_ok(monkeypatch):
    ing, embedder = make_ingestor(monkeypatch, returns("Theory exams start 5 Jan 2027. " * 20, pages=3))
    ing.ingest(LINK)
    saved = ing.store.saved[0]
    assert saved["status"] == "ok" and saved["num_pages"] == 3
    assert saved["chunks"][0].startswith("Notice: LLB Datesheet (dated 2026-09-28)")
    assert embedder.calls == 1 and ing.stats["stored"] == 1


def test_scanned_pdf_is_saved_title_only_with_single_chunk(monkeypatch):
    ing, _ = make_ingestor(monkeypatch, raises(NoTextError("no extractable text", 4)))
    ing.ingest(LINK)
    saved = ing.store.saved[0]
    assert saved["status"] == "title_only" and saved["num_pages"] == 4
    assert len(saved["chunks"]) == 1 and "LLB Datesheet" in saved["chunks"][0]
    assert "scanned image" in saved["chunks"][0] and saved["error"]
    assert ing.stats["title_only"] == 1


def test_download_failure_on_new_url_is_recorded(monkeypatch):
    ing, embedder = make_ingestor(monkeypatch, raises(ScrapeError("download failed: 404")))
    ing.ingest(LINK)
    assert ing.store.failures == ["download failed: 404"]
    assert embedder.calls == 0 and ing.stats["failed"] == 1


def test_failed_refresh_keeps_existing_good_copy(monkeypatch):
    store = FakeStore({LINK.url: KnownNotice(1, "ok", "h", utcnow() - timedelta(days=10))})
    ing, _ = make_ingestor(monkeypatch, raises(ScrapeError("timeout")), store)
    ing.ingest(LINK)
    assert store.failures == [] and store.saved == []


def test_unchanged_content_is_touched_not_reembedded(monkeypatch):
    fetch = returns("Same notice text, long enough to be real content. " * 5)
    ing, embedder = make_ingestor(monkeypatch, fetch)
    ing.ingest(LINK)  # first time: stored
    ing.ingest(LINK)  # weekly recompute: same hash
    assert embedder.calls == 1
    assert ing.store.touched == [LINK.url] and ing.stats["unchanged"] == 1


def test_changed_content_is_reembedded_as_update(monkeypatch):
    store = FakeStore({LINK.url: KnownNotice(1, "ok", "old-hash", utcnow() - timedelta(days=10))})
    ing, embedder = make_ingestor(monkeypatch, returns("Revised datesheet text. " * 10), store)
    ing.ingest(LINK)
    assert embedder.calls == 1 and ing.stats["updated"] == 1


def test_unexpected_error_never_crashes_the_run(monkeypatch):
    ing, _ = make_ingestor(monkeypatch, raises(RuntimeError("supabase exploded")))
    ing.ingest(LINK)  # must not raise
    assert ing.stats["failed"] == 1 and LINK.url in ing.attempted


# --- embedder -------------------------------------------------------------------------------

class FakeOllama:
    def __init__(self, dim=config.EMBED_DIM):
        self.dim, self.batches = dim, []

    def embed(self, model, input):
        self.batches.append(input)
        return {"embeddings": [[0.1] * self.dim for _ in input]}


def make_embedder(client):
    embedder = Embedder.__new__(Embedder)
    embedder.model, embedder.host, embedder.client = "bge-m3", "local", client
    return embedder


def test_embedder_sends_text_as_is_and_batches(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 2)
    client = FakeOllama()
    vectors = make_embedder(client).embed_documents(["a", "b", "c"])
    assert len(vectors) == 3 and [len(b) for b in client.batches] == [2, 1]
    assert client.batches == [["a", "b"], ["c"]]  # bge-m3: no task prefix


def test_embedder_rejects_wrong_dimension():
    with pytest.raises(ValueError, match=str(config.EMBED_DIM)):
        make_embedder(FakeOllama(dim=768)).embed_documents(["a"])


# --- Cloudflare embedder (EMBED_PROVIDER=cloudflare) -------------------------------------------

class FakeCloudflareSession:
    def __init__(self, status=200, dim=config.EMBED_DIM):
        self.status, self.dim, self.calls, self.headers = status, dim, [], {}

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        body = (
            {"success": True, "result": {"data": [[0.1] * self.dim for _ in json["text"]]}}
            if self.status == 200
            else {"success": False, "errors": [{"code": 10000, "message": "Authentication error"}]}
        )
        return FakeResponse(json_module.dumps(body), status=self.status, headers={"content-type": "application/json"})


def cloudflare_embedder(monkeypatch, session):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    embedder = ingest.CloudflareEmbedder()
    embedder.session = session
    return embedder


def test_cloudflare_embedder_batches_to_the_bge_m3_endpoint(monkeypatch):
    monkeypatch.setattr(config, "CLOUDFLARE_EMBED_BATCH_SIZE", 2)
    session = FakeCloudflareSession()
    vectors = cloudflare_embedder(monkeypatch, session).embed_documents(["a", "b", "c"])
    assert len(vectors) == 3 and all(len(v) == config.EMBED_DIM for v in vectors)
    assert [c[1]["text"] for c in session.calls] == [["a", "b"], ["c"]]
    assert session.calls[0][0] == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/baai/bge-m3"


def test_cloudflare_embedder_raises_on_api_error(monkeypatch):
    with pytest.raises(RuntimeError, match="401"):
        cloudflare_embedder(monkeypatch, FakeCloudflareSession(status=401)).embed_documents(["a"])


def test_cloudflare_embedder_requires_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    with pytest.raises(SystemExit, match="CLOUDFLARE_API_TOKEN"):
        ingest.CloudflareEmbedder()


def test_make_embedder_picks_provider_from_env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("EMBED_PROVIDER", "cloudflare")
    assert isinstance(ingest.make_embedder(), ingest.CloudflareEmbedder)
    monkeypatch.setenv("EMBED_PROVIDER", "ollama")
    assert isinstance(ingest.make_embedder(), ingest.Embedder)
