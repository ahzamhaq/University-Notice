-- Switch embeddings to bge-m3, step 2 of 2. Run ONLY after scripts/reembed_bge_m3.py has
-- filled every row and the retrieval tests passed against match_notice_chunks_m3.
-- Replaces the old 768-dim column with the new 1024-dim one and repoints match_notice_chunks.

do $$
begin
  if exists (select 1 from notice_chunks where embedding_m3 is null) then
    raise exception 'Some chunks have no bge-m3 embedding yet; run scripts/reembed_bge_m3.py first.';
  end if;
end $$;

drop function if exists match_notice_chunks_m3 (vector, int, float, int, float, float);
drop function if exists set_chunk_embeddings_m3 (jsonb);
drop function if exists match_notice_chunks (vector, int, float, int, float, float);

alter table notice_chunks drop column embedding;
alter table notice_chunks rename column embedding_m3 to embedding;
alter table notice_chunks alter column embedding set not null;

create or replace function match_notice_chunks (
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
