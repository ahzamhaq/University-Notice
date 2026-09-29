-- University Notice RAG schema. Run once in the Supabase SQL editor (safe to re-run).
-- Embedding size 1024 matches bge-m3 (config.EMBED_DIM); change it everywhere if you switch models.
-- Existing databases created with 768-dim nomic vectors: apply supabase/migrations/002 and 003 instead.

create extension if not exists vector;

-- One row per PDF URL. status: 'ok' = full text indexed; 'title_only' = scanned PDF,
-- only the title is indexed; 'failed' = download/parse error, kept so it isn't retried
-- until updated_at is REFRESH_DAYS old.
create table if not exists notices (
  id           bigserial primary key,
  url          text not null unique,
  title        text,
  notice_date  date,
  source_page  text,
  status       text not null default 'ok' check (status in ('ok', 'title_only', 'failed')),
  error        text,
  content_hash text,
  num_pages    int,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create index if not exists notices_updated_at_idx on notices (updated_at);

create table if not exists notice_chunks (
  id          bigserial primary key,
  notice_id   bigint not null references notices (id) on delete cascade,
  chunk_index int not null,
  content     text not null,
  embedding   vector(1024) not null,
  update_date timestamptz not null default now(),
  unique (notice_id, chunk_index)
);

-- No HNSW index on purpose: big scanned-list PDFs produce hundreds of near-identical
-- chunks that trap the approximate search (it missed the best matches in testing).
-- An exact scan is only milliseconds at this scale (tens of thousands of chunks).
drop index if exists notice_chunks_embedding_idx;

-- Exact nearest-neighbour search used by lib/rag.ts.
-- * At most max_per_notice chunks come from any one notice, so one huge PDF can't fill every slot.
-- * score = similarity + a recency boost that halves every recency_half_life_days, so a new
--   notice beats an old one with nearly the same match, while a clearly better old match still wins.
--   (0.065 / 180 days was tuned on real bge-m3 queries; the threshold still applies to raw similarity.)
drop function if exists match_notice_chunks (vector, int, float);
drop function if exists match_notice_chunks (vector, int, float, int);
create or replace function match_notice_chunks (
  query_embedding        vector(1024),
  match_count            int   default 5,
  match_threshold        float default 0.45,
  max_per_notice         int   default 1,
  recency_weight         float default 0.065,
  recency_half_life_days float default 180
)
returns table (
  id          bigint,
  notice_id   bigint,
  content     text,
  similarity  float,
  score       float,
  title       text,
  url         text,
  notice_date date,
  update_date timestamptz
)
language sql stable
as $$
  with scored as (
    select c.id, c.notice_id, c.content, c.update_date,
           1 - (c.embedding <=> query_embedding) as similarity
    from notice_chunks c
  ),
  ranked as (
    select s.*,
           row_number() over (partition by s.notice_id order by s.similarity desc) as rank_in_notice
    from scored s
    where s.similarity > match_threshold
  )
  select
    r.id, r.notice_id, r.content, r.similarity,
    r.similarity + coalesce(
      recency_weight * power(0.5, greatest(current_date - n.notice_date, 0) / recency_half_life_days),
      0
    ) as score,
    n.title, n.url, n.notice_date, r.update_date
  from ranked r
  join notices n on n.id = r.notice_id
  where r.rank_in_notice <= max_per_notice
  order by score desc
  limit match_count;
$$;

-- Lock the tables down: with RLS on and no policies, only the service-role key
-- (used server-side by the Python scripts and the Next.js API route) can read or write.
alter table notices enable row level security;
alter table notice_chunks enable row level security;
