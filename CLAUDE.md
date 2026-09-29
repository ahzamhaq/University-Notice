# CLAUDE.md

A RAG (retrieval-augmented generation) system for GGSIPU university notices. It scrapes notice PDFs from ipu.ac.in, embeds them locally with Ollama, stores the vectors in Supabase (pgvector), and answers questions through a Next.js UI.

## Architecture

```
ipu.ac.in pages ──> scripts/scraper.py ──> scripts/embed_and_store.py ──> Supabase (pgvector)
                    (crawl, download,       (chunk, embed via Ollama,          │
                     extract PDF text)       upsert with update_date)          │
                                                                               ▼
components/SearchNotices.tsx ──> app/api/query/route.ts ──> lib/rag.ts ──> Ollama (embed + chat)
        (React UI)                (rate limit, validation)   (retrieve + generate)
```

- **Ingestion is Python**, and **serving is TypeScript/Next.js**. They share only the database schema (`supabase/schema.sql`) and the env vars.
- **Everything model-related runs locally through Ollama.** Embeddings use `nomic-embed-text` (768 dims) and answers use a local chat model (`OLLAMA_CHAT_MODEL`, default `llama3.2`). Don't add hosted LLM APIs.
- nomic-embed-text needs task prefixes: `search_document: ` when indexing and `search_query: ` when querying. Both sides must stay in sync.

## Key files

| Path | Role |
| --- | --- |
| `config.py` | All ingestion settings (URLs, chunking, limits, delays). Scripts import it from the repo root. |
| `urls.txt` | Optional extra seed URLs, one per line, merged with `NOTICE_URLS`, so new pages can be added without code changes. |
| `scripts/scraper.py` | Crawls seed pages (pagination + same-site recursion), finds PDF links, downloads and extracts text. Runs standalone as a dry run. |
| `scripts/embed_and_store.py` | Main ingestion entry point. Uses the scraper, chunks the text, embeds it and writes to Supabase. |
| `supabase/schema.sql` | Tables `notices` and `notice_chunks`, the HNSW index and the `match_notice_chunks` RPC. |
| `lib/rag.ts` | Query embedding, vector search via RPC, prompt building and the Ollama chat call. |
| `lib/rate-limit.ts`, `lib/logger.ts` | In-memory per-IP limiter and console+file logging for the API. |
| `app/api/query/route.ts` | POST endpoint `{ question }` → `{ answer, sources }`. |
| `components/SearchNotices.tsx` | Client component with the search box, answer and source list. |

## Ingestion rules (keep these invariants)

- **Politeness:** wait `REQUEST_DELAY` (2s) between every HTTP request to ipu.ac.in and honour `robots.txt`.
- **Dedup:** check whether a PDF URL is already in `notices` *before* downloading it. Skip it unless its `updated_at` is older than `REFRESH_DAYS` (7), in which case re-download and re-embed it (the weekly recompute).
- **Size cap:** skip PDFs over `PDF_SIZE_LIMIT` (100MB). Check `Content-Length` first, then enforce the limit while streaming.
- **Failures never crash the run.** A PDF that fails to download or parse is logged and recorded with `status='failed'` and an `error` message.
- **Scanned PDFs** (they parse, but have no text layer; `NoTextError`) are stored with `status='title_only'` and a single chunk made of the title, date and a note that only the title is available. OCR is a possible later upgrade.
- **Chunking:** 500 tokens with 50 overlap (tiktoken `cl100k_base` as the token counter), via LangChain's `RecursiveCharacterTextSplitter`.
- Every chunk row carries `update_date`. Re-embedding a notice deletes its old chunks first.

## Retrieval rules

- `match_notice_chunks` does an **exact** scan and returns at most `max_per_notice` chunks per notice (`lib/rag.ts` asks for 5 notices with 1 chunk each). Don't add an HNSW/IVFFlat index back: huge scanned-list PDFs create hundreds of near-identical chunks that trapped the approximate search, which then missed the best matches. Exact search is only milliseconds at this scale.
- **Ranking** is `score = similarity + 0.05 × 0.5^(age_days / 180)`. That lets a new notice beat an old one with nearly the same match, while a clearly better old match still wins. The weight was tuned on real queries (0.08 started favouring less relevant new notices). `tests/test_live.py` has regression checks for both behaviours.
- Sources and prompt excerpts keep the **score order** from the database, so `[1]` is the best match. Recency already comes from the boost. Do not also sort by date: that pushed a clearly best (older, title-only) notice to `[5]`, and the 3B model then ignored it.
- **Speed:** Ollama runs on the CPU here (about 15 prompt tokens/s and 5 output tokens/s, on an i3-N305). So the prompt is kept to about 1.5k tokens (excerpts trimmed to 1,000 chars), with `num_predict: 350`, `keep_alive: 30m` and a 300s timeout. The system prompt stays the same all day so Ollama can reuse its cached prefix.
- **Streaming:** `POST /api/query` with `Accept: application/x-ndjson` streams `sources` (after about 3s), then `token` events, then `done` or `error`. Without that header it returns plain JSON. When the client disconnects, the request is aborted so the model stops generating. A timeout returns 504 and is not reported as "Ollama unreachable".

## API rules

- Rate limit: 50 requests per minute per IP (sliding window, in memory, per server instance).
- Log all errors to the console and to `logs/api.log`. Python logs go to `logs/ingest.log`.
- `SUPABASE_KEY` is the service-role key. Use it only on the server (API route and Python) and never in client components.

## Commands

```bash
pip install -r requirements.txt
python scripts/scraper.py --max-pages 1 --max-pdfs 5   # dry run: crawl + extract to data/extracted.jsonl, no DB
python scripts/embed_and_store.py --max-pdfs 10        # quick ingestion test
python scripts/embed_and_store.py                      # full ingestion (schedule weekly)
npm install && npm run dev                             # UI at http://localhost:3000
npm run build                                          # also type-checks
venv/Scripts/python -m pytest -m "not live"            # offline Python unit tests
npm test                                               # offline TypeScript tests (vitest)
API_URL=http://localhost:3000 venv/Scripts/python -m pytest -m live   # read-only checks against real Ollama/Supabase/site/API
```

## Testing

- `tests/test_scraper.py` and `tests/test_ingest.py` are offline: they use fakes from `tests/conftest.py` (`FakeSession`, `FakeResponse`, generated PDFs). Never hit the network in unit tests.
- `tests/*.test.ts` run under vitest. With vitest 5, don't call `mockReset()` or `mockClear()` on a mock that later throws, because it fails the test even when the code catches the error. Compare call counts instead.
- `tests/test_live.py` (marker `live`) is read-only. It checks data integrity, retrieval quality on known notices and the API end to end. Update the expected titles if old notices are removed.

## Known site quirks

- The listing pages (`notices.php`, `dsw_sports.php`) use a plain `<table>` with a `Next »` link (`?page=N&limit=25`). `notices.php` has 800+ pages, which is why pagination is capped.
- The site menus contain unrelated PDF links (the `dropdown-item` class). The scraper skips them.
- Some rows appear twice, and filenames can contain spaces. URLs are normalised with `requote_uri` before dedup.
- Many notices (about 3 in 4 in a sample) are scanned PDFs with no text layer. They are indexed by title only (`title_only`).
- Stack versions: Next.js 16, React 19, TypeScript 7.

## Conventions

- Python 3.10+ with type hints and the `logging` module (no `print` in library code).
- TypeScript strict mode and the Next.js App Router. The API route runs on the Node runtime because it needs `fs` for logging.
- Keep secrets in `.env` (git-ignored). When you add a new env var, also add it to `.env.example`.
