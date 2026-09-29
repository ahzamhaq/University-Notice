# IPU Notice Search

**Live demo: <https://university-notice.vercel.app>**

**Official university website: <https://www.ipu.ac.in/>**

Ask questions about GGSIPU (Guru Gobind Singh Indraprastha University) notices in plain English and get a short answer that cites the official PDFs. This is a retrieval-augmented generation (RAG) system:
- A Python pipeline scrapes and indexes the notice PDFs.
- Supabase (pgvector) stores the embeddings.
- A Next.js app searches them and streams an answer from an LLM.

It runs **fully offline** on your own machine with Ollama, or **deployed for free** on Vercel with Cloudflare Workers AI.

| | |
| --- | --- |
| Notices indexed | **1,458** from **25** IPU pages |
| Searchable chunks | **7,858** (`bge-m3`, 1,024-dim vectors) |
| Answer time (deployed) | sources in about **4s**, full answer in about **6s** |
| Automated tests | **161**: 68 Python unit, 45 TypeScript, 48 live |

## Features

- **Answers with citations.** The model reads the 5 best-matching notices and cites them as `[1]`–`[5]`. Each citation links to the original PDF.
- **Streaming.** Matching notices appear as soon as the search finishes, and the answer then streams in word by word.
- **Up to 20 matching notices.** The 5 cited notices are shown first, and **Load more documents** reveals the rest.
- **Newest first or Best match.** Rows are numbered in the order shown, and cited rows carry a `cited [n]` tag that matches the answer.
- **Recency-aware ranking.** A tuned recency boost ranks this year's notices above near-identical older ones, while a clearly better older match still wins.
- **Honest about scanned PDFs.** Many IPU notices are scanned images. They are indexed by title and date, and the answer says the details are in the PDF instead of inventing them.
- **Filters unrelated questions.** Off-topic questions such as "pizza recipe" match nothing, instead of pulling in random notices.
- **Rate limited and safe.** The API allows 50 requests per minute per IP. The database is locked with row-level security, and no keys reach the browser.

## Architecture

```
                 ingestion (runs on your machine)                              query (Vercel or local)
ipu.ac.in ─► scripts/scraper.py ─► scripts/embed_and_store.py ─► Supabase ◄─ lib/rag.ts ◄─ /api/query ◄─ SearchNotices.tsx
              crawl + PDF text      chunk + embed (bge-m3)        pgvector      search + LLM    rate limit    React UI
```

| Piece | Local mode (default, offline) | Hosted mode (Vercel) |
| --- | --- | --- |
| Embeddings | Ollama `bge-m3` | Cloudflare Workers AI `@cf/baai/bge-m3`, the same model with identical vectors (cosine 1.0000 in tests) |
| Answers | Ollama `llama3.2` | Workers AI `@cf/meta/llama-3.1-8b-instruct-fp8-fast` |
| Database | Supabase (or a local Supabase in Docker) | Supabase |

The mode is chosen by environment variables (`EMBED_PROVIDER`, `LLM_PROVIDER` = `ollama` or `cloudflare`), so the same code runs in both.

**Tech stack:** Python (BeautifulSoup, PyPDF2, LangChain text splitters, tiktoken) · TypeScript, Next.js 16, React 19 · Supabase / PostgreSQL / pgvector · Ollama · Cloudflare Workers AI · pytest, Vitest

## Project structure

| Path | What it does |
| --- | --- |
| `scripts/scraper.py` | Crawls the notice pages (pagination, robots.txt, 2s delay) and downloads and extracts PDF text |
| `scripts/embed_and_store.py` | Main ingestion: chunk, embed and save to Supabase, with a weekly refresh |
| `scripts/reembed_bge_m3.py` | One-off re-embedding used when switching embedding models |
| `config.py`, `urls.txt` | Crawl settings and the list of pages to scrape |
| `supabase/schema.sql` | Tables, row-level security and the `match_notice_chunks` search function |
| `supabase/migrations/` | Migration from 768-dim `nomic-embed-text` to 1,024-dim `bge-m3` |
| `lib/rag.ts` | Search, prompt building and streaming from Ollama or Cloudflare |
| `app/api/query/route.ts` | API: input validation, rate limiting, JSON or streamed (NDJSON) responses |
| `components/SearchNotices.tsx` | The search UI |
| `tests/` | Python and TypeScript test suites |

## Run it locally

**Prerequisites:** Python 3.10+, Node.js 20+, [Ollama](https://ollama.com/download), and a free [Supabase](https://supabase.com) project.

**1. Models**

```bash
ollama pull bge-m3             # embeddings (1024 dims, ~1.2GB)
ollama pull llama3.2           # answers (~2GB); any Ollama chat model works, see OLLAMA_CHAT_MODEL
```

**2. Database.** In your Supabase project, open **SQL Editor** and run all of [`supabase/schema.sql`](supabase/schema.sql). Then copy the **Project URL** and the **service_role** key from *Project Settings → API*.

**3. Environment**

```bash
cp .env.example .env           # fill in SUPABASE_URL and SUPABASE_KEY
```

The tables have row-level security with no policies, so only the service_role key can read them. That key is used only by the Python scripts and the Next.js API route, never in the browser.

**4. Install**

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
npm install
```

**5. Ingest notices**

```bash
python scripts/embed_and_store.py --max-pdfs 10    # quick test
python scripts/embed_and_store.py                  # full run
```

Each PDF is logged as `STORED`, `TITLE` (a scanned PDF, indexed by title only), `UNCHANGED` or `FAILED` (with the reason), to the console and to `logs/ingest.log`. To embed much faster than a CPU can, set `EMBED_PROVIDER=cloudflare` together with your Cloudflare credentials; it gives the same vectors.

**6. Start the app**

```bash
npm run dev                    # http://localhost:3000
```

## Deploy to Vercel (free)

Vercel can't run Ollama, so the deployed site uses Cloudflare Workers AI for the query side. The free plan's 10,000 daily "neurons" cover roughly **600 questions a day**, at about 15 neurons per question.

1. **Cloudflare:** in *Manage account → Account API tokens*, create a token with only the **Workers AI** permissions. Note your Account ID, the 32-character code in the dashboard URL.
2. **Vercel:** import the GitHub repo at <https://vercel.com/new>. Next.js is detected automatically.
3. **Environment variables:**

   | Name | Value |
   | --- | --- |
   | `SUPABASE_URL` | your project URL (nothing after `.supabase.co`) |
   | `SUPABASE_KEY` | service_role key |
   | `EMBED_PROVIDER` | `cloudflare` |
   | `LLM_PROVIDER` | `cloudflare` |
   | `CLOUDFLARE_ACCOUNT_ID` | your account ID |
   | `CLOUDFLARE_API_TOKEN` | your token |

4. **Deploy.** Every push to `main` redeploys automatically. New notices need no redeploy: run the ingestion locally, and the site reads the same database.

## How it works

**Scraping** (`scripts/scraper.py`)
- Reads the notice table on each page listed in `NOTICE_URLS` / [`urls.txt`](urls.txt). Titles and dates come from the table row.
- When a link just says "PDF", the title is taken from the row's other cells. Both `dd-mm-yyyy` and `dd-mm-yy` dates are parsed.
- Follows `Next »` pagination, up to 5 pages per listing by default. It stops early when a whole page is already stored; `--all-pages` walks the full history. Only the listed pages are crawled (`CRAWL_MAX_DEPTH = 0`).
- **Polite:** it honours `robots.txt` (including any `Crawl-delay`) and waits 2 seconds between requests, timed with a high-resolution clock.
- **Deduplicated:** stored URLs are skipped *before* downloading. PDFs over 100MB are rejected, both from their declared size and while streaming.
- **Scanned PDFs** (no text layer) are stored as `title_only`. Download or parse errors are stored as `failed` and never stop the run.

**Ingestion** (`scripts/embed_and_store.py`)
- Text is split into chunks of 500 tokens with a 50-token overlap (LangChain `RecursiveCharacterTextSplitter`, counted with tiktoken). Each chunk is prefixed with the notice title and date.
- **Weekly refresh:** notices older than 7 days are downloaded again. If the text is unchanged, only `update_date` is bumped; if it changed, the notice is re-embedded. Schedule the script weekly with Task Scheduler or cron.

**Search and answers** (`lib/rag.ts`)
- The question is embedded with `bge-m3`. `match_notice_chunks` then does an **exact** cosine search, keeping the best chunk per notice, dropping anything below similarity **0.45**, and returning up to **20** notices.
- **Ranking:** `score = similarity + 0.065 × 0.5^(age_days / 180)`. This was tuned on real questions: a weaker boost let 2017 election notices beat this year's, and a stronger one pushed down correct older answers.
- The model reads only the **top 5** notices. Their excerpts are trimmed to about 1,000 characters to keep answers fast, and scanned notices are clearly marked as having no details, so the model doesn't borrow dates from other notices.
- **Why no HNSW index:** large scanned-list PDFs create hundreds of near-identical chunks that trapped the approximate search, which then missed the best matches. At this scale an exact scan takes only milliseconds.

**API** (`POST /api/query` with `{ "question": "..." }`)
- It returns JSON `{ answer, sources }` by default. With `Accept: application/x-ndjson`, it streams `sources`, then `token` events, then `done` or `error`.
- Each source has `title`, `url`, `noticeDate`, `similarity`, `ref` (its relevance rank) and `cited` (whether the model read it).
- It allows 50 requests per minute per IP (`429` with `Retry-After`). The limiter lives in memory, so on serverless it applies per instance.

## Tests

```bash
pip install -r requirements-dev.txt
venv/Scripts/python -m pytest -m "not live"                     # 68 offline Python tests (scraper, ingestion)
npm test                                                        # 45 offline TypeScript tests (API, RAG, rate limit)
API_URL=http://localhost:3000 venv/Scripts/python -m pytest -m live
```

The **live** suite is read-only and checks the real Ollama, Supabase, ipu.ac.in and API. It covers:
- data integrity (duplicates, orphan chunks, embedding size)
- row-level security
- search quality on known questions, including the recency and off-topic cases
- that Cloudflare `bge-m3` produces the same vectors as local `bge-m3`
- the API end to end, including streaming

Point `API_URL` at the Vercel URL to check production.

## Adding notice pages

Add a line to [`urls.txt`](urls.txt), with no code changes, or pass `--url` for a single run:

```bash
python scripts/embed_and_store.py --url https://www.ipu.ac.in/some_page.php
```

Other options: `--max-pages N`, `--all-pages`, `--depth N`, `--max-pdfs N` and `--no-refresh`. The defaults are in [`config.py`](config.py).

## Known limitations

- **Scanned PDFs:** about half of the notices have no text layer, so only their title and date are searchable. Adding OCR is the next improvement.
- **Research pages** (`rnc_*`) are paused in `urls.txt`. Their existing notices stay in the database, but 52 of them are still titled "PDF" until those pages are crawled again.
- **Free tiers:** Cloudflare allows about 600 questions a day. Supabase pauses a free project after about a week without activity (click **Restore**).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Cannot reach Ollama` / 503 | Start `ollama serve` and check `OLLAMA_ENDPOINT` (local mode) |
| `The embedding model returned 768 dimensions…` | Set `OLLAMA_EMBED_MODEL=bge-m3` (an old `nomic-embed-text` setting) |
| `… is not set on the server` | Add the named variable in Vercel (*Settings → Environment Variables*), then redeploy |
| `Vector search failed` | Check `SUPABASE_URL` / `SUPABASE_KEY`, and that `supabase/schema.sql` has been run |
| `service is busy (rate limit)` | The Cloudflare free daily allowance is used up; it resets the next day |
| Upgrading a database created with 768-dim vectors | Run `supabase/migrations/002`, then `scripts/reembed_bge_m3.py`, then `003` |
