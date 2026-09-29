-- Switch embeddings from nomic-embed-text (768 dims) to bge-m3 (1024 dims), step 1 of 2.
-- Adds a parallel column so the live search keeps working while it is filled by
-- scripts/reembed_bge_m3.py and tested. Nothing existing is changed or removed.
-- Don't run ingestion between step 1 and step 2.

alter table notice_chunks add column if not exists embedding_m3 vector(1024);

-- Bulk update used by the re-embed script. payload: [{"id": 1, "embedding": [..1024 floats..]}, ...]
create or replace function set_chunk_embeddings_m3 (payload jsonb)
returns int
language sql
as $$
  with rows as (
    select (r->>'id')::bigint as id, (r->'embedding')::text::vector(1024) as emb
    from jsonb_array_elements(payload) as r
  ),
  updated as (
    update notice_chunks c set embedding_m3 = rows.emb
    from rows
    where c.id = rows.id
    returning 1
  )
  select count(*)::int from updated;
$$;

-- Same ranking as match_notice_chunks, but over the new column, for testing before the swap.
create or replace function match_notice_chunks_m3 (
  query_embedding        vector(1024),
  match_count            int   default 5,
  match_threshold        float default 0.45,
  max_per_notice         int   default 1,
  recency_weight         float default 0.05,
  recency_half_life_days float default 180
)
returns table (
  id bigint, notice_id bigint, content text, similarity float, score float,
  title text, url text, notice_date date, update_date timestamptz
)
language sql stable
as $$
  with scored as (
    select c.id, c.notice_id, c.content, c.update_date,
           1 - (c.embedding_m3 <=> query_embedding) as similarity
    from notice_chunks c
    where c.embedding_m3 is not null
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

-- Only the service-role key (server side) may call these.
revoke execute on function set_chunk_embeddings_m3 (jsonb) from public, anon, authenticated;
revoke execute on function match_notice_chunks_m3 (vector, int, float, int, float, float)
  from public, anon, authenticated;
