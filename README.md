# University Notice RAG

Ask questions about GGSIPU notices in plain English. The pipeline scrapes the notice PDFs from
[ipu.ac.in/notices.php](https://www.ipu.ac.in/notices.php) and
[dsw_sports.php](https://www.ipu.ac.in/dsw_sports.php), embeds them **locally** with Ollama, stores the
vectors in Supabase (pgvector), and answers questions through a Next.js UI, citing the source PDFs.

```
scraper.py ─► embed_and_store.py ─► Supabase pgvector ◄─ lib/rag.ts ◄─ /api/query ◄─ SearchNotices.tsx
                  (Ollama: bge-m3)                           (Ollama: llama3.2)
```

## Prerequisites

- Python 3.10+
- Node.js 20+
- [Ollama](https://ollama.com/download)
- A Supabase project (the free tier is fine), or a local one (see [Fully offline](#fully-offline))

## Setup

### 1. Install Ollama and pull the models

- **Windows / macOS:** download the installer from <https://ollama.com/download>
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`

```bash
ollama pull bge-m3             # embeddings (1024 dims, ~1.2GB)
ollama pull llama3.2           # answer generation (~2GB); any chat model works, see OLLAMA_CHAT_MODEL
ollama serve                   # keep running in another terminal (the desktop app starts it automatically)
```

### 2. Create the database

1. Create a project at <https://supabase.com>.
2. Open **SQL Editor** and run the whole of [`supabase/schema.sql`](supabase/schema.sql). It enables pgvector and
   creates the `notices` and `notice_chunks` tables, an HNSW index and the `match_notice_chunks` search function.
3. Copy the **Project URL** and the **service_role** key from *Project Settings → API*.

### 3. Configure the environment

```bash
cp .env.example .env
```

Fill in `SUPABASE_URL` and `SUPABASE_KEY` (the service_role key). The tables have row-level security
switched on with no policies, so only this key can read them. It is used only by the Python scripts and the
Next.js API route, and is never sent to the browser.

### 4. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
npm install
```

### 5. Ingest the notices

```bash
python scripts/embed_and_store.py --max-pdfs 10   # quick test first
python scripts/embed_and_store.py                 # full run
```

Progress is printed and also written to `logs/ingest.log`. Each PDF is logged as `STORED`, `UNCHANGED` or
`TITLE` (a scanned PDF, indexed by title only) or `FAILED` (with the reason), and the run ends with a summary. To try the scraper without a database, run
`python scripts/scraper.py --max-pages 1 --max-pdfs 5`, which writes to `data/extracted.jsonl`.

### 6. Run the app

```bash
npm run dev
```

Open <http://localhost:3000>.

## How it works

**Scraping** (`scripts/scraper.py`)
- Reads the notice table on each seed page. Each PDF link gets its title and upload date from the table row.
  Menu and header links are ignored.
- Follows the `Next »` pagination up to `MAX_PAGES_PER_LISTING` pages per listing (notices.php has 800+ pages,
  newest first). It stops early once a whole page contains only PDFs that are already stored.
- **Recursive:** it follows linked sub-pages up to `CRAWL_MAX_DEPTH` levels, but only those on `ALLOWED_DOMAINS`
  whose URL matches `FOLLOW_LINK_PATTERNS`.
- **Polite:** it honours `robots.txt` (including any `Crawl-delay`) and waits 2 seconds between requests.
- **Duplicates** are skipped *before* downloading: stored URLs are loaded from Supabase at startup.
- **Oversized PDFs** (over 100MB) are rejected from `Content-Length`, with the limit also enforced while streaming.
- **Failures** (download errors, corrupt or encrypted PDFs, scanned PDFs with no text layer) are logged and
  saved with `status='failed'`, and never stop the run.

**Embedding** (`scripts/embed_and_store.py`)
- Splits text into chunks of 500 tokens with a 50-token overlap (LangChain `RecursiveCharacterTextSplitter`,
  counted with tiktoken). Each chunk is prefixed with the notice title and date.
- Embeds the chunks with `bge-m3` (1024 dims) through the local Ollama server, or on Cloudflare Workers AI with `EMBED_PROVIDER=cloudflare` (same vectors, much faster).
- Every chunk row has an `update_date`.

**Weekly recompute:** stored notices whose `updated_at` is more than `REFRESH_DAYS` (7) old are downloaded again
on the next run. If the text is unchanged, only `update_date` is bumped. If it has changed, the notice is
re-chunked and re-embedded. Failed PDFs are retried on the same weekly schedule. Schedule the script weekly:

```bash
# Windows (Task Scheduler), every Sunday at 03:00; adjust the paths
schtasks /Create /TN "NoticeRAG" /SC WEEKLY /D SUN /ST 03:00 /TR "\"C:\path\to\venv\Scripts\python.exe\" \"C:\path\to\scripts\embed_and_store.py\""
```

```bash
# macOS / Linux (crontab -e)
0 3 * * 0  cd /path/to/project && venv/bin/python scripts/embed_and_store.py
```

**Querying** (`lib/rag.ts`, `app/api/query/route.ts`)
- `POST /api/query` with `{ "question": "..." }` returns `{ "answer": "...", "sources": [{ title, url, noticeDate, similarity }] }`.
- The question is embedded with the `search_query:` prefix, and the 6 nearest chunks (cosine similarity above 0.3) are
  fetched through `match_notice_chunks`. The chunks are grouped by notice, and the local chat model answers
  with numbered citations.
- The endpoint is **rate limited** to 50 requests per minute per IP and returns `429` with `Retry-After` when
  the limit is hit. The limiter lives in memory, so each server process has its own limit.
- Errors are logged as JSON lines to the console and to `logs/api.log`.

## Deploying to Vercel

Vercel can't run Ollama, so the deployed site uses **Cloudflare Workers AI** (free plan) for the query side. Ingestion still runs on your machine.

| Piece | Local (default, offline) | Deployed |
| --- | --- | --- |
| Question embedding | Ollama `bge-m3` | Workers AI `@cf/baai/bge-m3` (same model, identical vectors: cosine 1.0000 in tests) |
| Answers | Ollama `llama3.2` | Workers AI `@cf/meta/llama-3.1-8b-instruct-fp8-fast` |
| Database | Supabase | Supabase (same project) |

The free plan gives 10,000 "neurons" a day. One question costs about 15 (about 1,500 tokens in and 250 out), so it covers roughly 600 questions a day.

1. Create a Cloudflare account. Under **Manage account → Account API tokens**, create a token with only the **Workers AI** permissions, and note your Account ID (the 32-character code in the dashboard URL).
2. Import the GitHub repo at <https://vercel.com/new>. The framework is detected as Next.js.
3. Add these environment variables:

   | Name | Value |
   | --- | --- |
   | `SUPABASE_URL` | your project URL |
   | `SUPABASE_KEY` | service_role key |
   | `EMBED_PROVIDER` | `cloudflare` |
   | `LLM_PROVIDER` | `cloudflare` |
   | `CLOUDFLARE_ACCOUNT_ID` | your account ID |
   | `CLOUDFLARE_API_TOKEN` | your token |

4. Deploy. To add notices later, run `python scripts/embed_and_store.py` locally. The site reads the same database, so there's no need to redeploy. To embed much faster than your CPU can, add `EMBED_PROVIDER=cloudflare` to your local `.env`; it gives the same vectors.

Notes: the per-IP rate limiter lives in memory, so on serverless it applies per instance.

## Tests

```bash
pip install -r requirements-dev.txt
venv/Scripts/python -m pytest -m "not live"      # offline Python unit tests (scraper, ingestion)
npm test                                         # offline TypeScript tests (rate limit, API route, RAG)
```

The live checks run against your real Ollama, Supabase and ipu.ac.in, and against the API if it's running. They are read-only:

```bash
API_URL=http://localhost:3000 venv/Scripts/python -m pytest -m live
```

## Adding more notice pages

You don't need to change any code. Either add a line to [`urls.txt`](urls.txt):

```
https://www.ipu.ac.in/some_other_notices.php
```

…or pass seed URLs for a single run:

```bash
python scripts/embed_and_store.py --url https://www.ipu.ac.in/some_page.php
```

Other options: `--max-pages N`, `--depth N`, `--max-pdfs N` and `--no-refresh`. All defaults are in
[`config.py`](config.py).

## Fully offline

Embeddings and answers already run locally through Ollama. To take Supabase offline too, run it locally with
Docker and the [Supabase CLI](https://supabase.com/docs/guides/local-development):

```bash
npx supabase init
npx supabase start          # prints the API URL (http://127.0.0.1:54321) and the service_role key
```

Point `SUPABASE_URL` and `SUPABASE_KEY` at the local instance, then run `schema.sql` in the local Studio
(<http://127.0.0.1:54323>). You need internet access only while scraping new notices.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Cannot reach Ollama` / API returns 503 | Start `ollama serve`, and check that `OLLAMA_ENDPOINT` is correct |
| `model ... not available` | `ollama pull bge-m3` (or the chat model you set) |
| `Could not read the notices table` | Run `supabase/schema.sql` in the SQL editor |
| Many `TITLE ... storing title only` lines | Many IPU notices are scanned images without a text layer. They are indexed by title and date only (`status='title_only'`). Adding OCR would be needed to index their full text |
| Answers say nothing was found | Check that ingestion stored chunks (`select count(*) from notice_chunks;`) |
| Switching embedding models | Update `EMBED_DIM` in `config.py` and `vector(768)` in `schema.sql`, then re-ingest |
