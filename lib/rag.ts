import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// Server-only RAG pipeline: embed the question with Ollama, find the best-matching notices
// in Supabase, then stream an answer from a local Ollama chat model.
//
// Speed matters because Ollama may run CPU-only (~15 prompt tokens/s): sources are emitted
// as soon as retrieval finishes, the answer is streamed token by token, and the prompt is
// kept small (one trimmed excerpt per notice).

const OLLAMA_ENDPOINT = (process.env.OLLAMA_ENDPOINT ?? "http://localhost:11434").replace(/\/+$/, "");
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL ?? "nomic-embed-text";
const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL ?? "llama3.2";

const MATCH_COUNT = 5; // notices shown as sources and given to the model
const MATCH_THRESHOLD = 0.3;
const MAX_CHUNKS_PER_NOTICE = 1;
const MAX_EXCERPT_CHARS = 1000; // ~250 tokens per notice keeps the prompt near 1.5k tokens
const EMBED_TIMEOUT_MS = 60_000;
const CHAT_TIMEOUT_MS = 300_000;
// Keep models loaded between questions so only the first one pays the load time.
const KEEP_ALIVE = "30m";
const CHAT_OPTIONS = { temperature: 0.2, num_ctx: 4096, num_predict: 350 };

export const NOT_FOUND_ANSWER = "I couldn't find any notices related to that question.";

export interface NoticeChunk {
  id: number;
  notice_id: number;
  content: string;
  similarity: number;
  score: number;
  title: string | null;
  url: string;
  notice_date: string | null;
  update_date: string;
}

export interface Source {
  title: string;
  url: string;
  noticeDate: string | null;
  similarity: number;
}

export interface RagResult {
  answer: string;
  sources: Source[];
}

/** Events streamed to the client: sources first, then answer tokens, then done. */
export type RagEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done" };

/** An error whose message is safe to show to the user; details go in `cause`. */
export class RagError extends Error {
  constructor(
    message: string,
    readonly status = 500,
    options?: { cause?: unknown },
  ) {
    super(message, options);
    this.name = "RagError";
  }
}

let supabase: SupabaseClient | null = null;

function getSupabase(): SupabaseClient {
  if (!supabase) {
    // The client adds /rest/v1 itself; a URL copied with that suffix causes PGRST125 errors.
    const url = process.env.SUPABASE_URL?.trim().replace(/\/(rest\/v1)?\/*$/, "");
    const key = process.env.SUPABASE_KEY?.trim();
    if (!url || !key) throw new RagError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY.");
    supabase = createClient(url, key, { auth: { persistSession: false } });
  }
  return supabase;
}

function withTimeout(timeoutMs: number, signal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([timeout, signal]) : timeout;
}

/** Map fetch/stream failures to user-facing errors. Client disconnects are rethrown as-is. */
function ollamaError(err: unknown, timeoutMs: number, signal?: AbortSignal): unknown {
  if (signal?.aborted || err instanceof RagError) return err;
  if (err instanceof Error && err.name === "TimeoutError") {
    return new RagError(
      `The local model took longer than ${timeoutMs / 1000}s to respond. It may still be loading; try again.`,
      504,
      { cause: err },
    );
  }
  return new RagError(`Could not reach Ollama at ${OLLAMA_ENDPOINT}. Is \`ollama serve\` running?`, 503, {
    cause: err,
  });
}

async function postOllama(
  route: string,
  body: { model: string } & Record<string, unknown>,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${OLLAMA_ENDPOINT}${route}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: withTimeout(timeoutMs, signal),
    });
  } catch (err) {
    throw ollamaError(err, timeoutMs, signal);
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new RagError(`Ollama returned ${res.status}. Has the model "${body.model}" been pulled?`, 502, {
      cause: detail,
    });
  }
  return res;
}

/** Yield complete lines from a streaming response body (Ollama streams NDJSON). */
async function* readLines(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) if (line.trim()) yield line;
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield buffer;
  } finally {
    reader.releaseLock();
  }
}

export async function embedQuery(question: string, signal?: AbortSignal): Promise<number[]> {
  const res = await postOllama(
    "/api/embed",
    // nomic-embed-text task prefix; documents are indexed with "search_document: ".
    { model: EMBED_MODEL, input: `search_query: ${question}`, keep_alive: KEEP_ALIVE },
    EMBED_TIMEOUT_MS,
    signal,
  );
  const data = (await res.json()) as { embeddings?: number[][] };
  const vector = data.embeddings?.[0];
  if (!vector?.length) throw new RagError("Ollama returned an empty embedding.", 502);
  return vector;
}

export async function retrieveChunks(question: string, signal?: AbortSignal): Promise<NoticeChunk[]> {
  const embedding = await embedQuery(question, signal);
  const { data, error } = await getSupabase().rpc("match_notice_chunks", {
    query_embedding: embedding,
    match_count: MATCH_COUNT,
    match_threshold: MATCH_THRESHOLD,
    max_per_notice: MAX_CHUNKS_PER_NOTICE,
  });
  if (error) {
    throw new RagError("Vector search failed. Has supabase/schema.sql been applied?", 502, { cause: error });
  }
  return (data ?? []) as NoticeChunk[];
}

interface NoticeGroup {
  source: Source;
  excerpts: string[];
}

/**
 * Group chunks by notice, keeping retrieval order. That order is the database's score
 * (similarity + recency boost), so recent notices already come first unless an older one
 * is a clearly better match. Re-sorting purely by date buried the best match in testing.
 */
export function groupByNotice(chunks: NoticeChunk[]): NoticeGroup[] {
  const groups = new Map<string, NoticeGroup>();
  for (const chunk of chunks) {
    let group = groups.get(chunk.url);
    if (!group) {
      group = {
        source: {
          title: chunk.title ?? "Untitled notice",
          url: chunk.url,
          noticeDate: chunk.notice_date,
          similarity: chunk.similarity,
        },
        excerpts: [],
      };
      groups.set(chunk.url, group);
    }
    group.excerpts.push(chunk.content);
    group.source.similarity = Math.max(group.source.similarity, chunk.similarity);
  }
  return [...groups.values()];
}

export function trimExcerpt(text: string, maxChars = MAX_EXCERPT_CHARS): string {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).replace(/\s+\S*$/, "") + " …";
}

// Written by scripts/embed_and_store.py (TITLE_ONLY_NOTE) for scanned PDFs.
const TITLE_ONLY_MARKER = "(Only the title of this notice is available";

function describeExcerpts(excerpts: string[]): string {
  if (excerpts.every((e) => e.includes(TITLE_ONLY_MARKER))) {
    // A small model given "only the title is available" tends to borrow details from other
    // notices; spell out that this notice has none.
    return "CONTENT: not available (scanned PDF). Only the title above is known; it has NO dates, times or venues.";
  }
  return "CONTENT:\n" + excerpts.map((e) => trimExcerpt(e)).join("\n...\n");
}

function buildMessages(question: string, groups: NoticeGroup[]) {
  const context = groups
    .map(
      (g, i) =>
        `[${i + 1}] TITLE: ${g.source.title}\nNOTICE DATE: ${g.source.noticeDate ?? "unknown"}\n` +
        describeExcerpts(g.excerpts),
    )
    .join("\n\n---\n\n");

  const today = new Date().toISOString().slice(0, 10);
  // Kept identical for a whole day so Ollama can reuse its cached prefix between questions.
  const system =
    "You answer questions about Guru Gobind Singh Indraprastha University (GGSIPU) notices. " +
    "Use ONLY the notice excerpts provided. They are listed most relevant first. " +
    "Cite the notices you use with their number, like [1]. " +
    "Each notice is separate: NEVER attach a date, time, venue or detail from one notice to another. " +
    "Some notices are scanned PDFs with no content: their title is still reliable evidence, but say " +
    "that the details are in the PDF and do not invent any. " +
    "Use NOTICE DATE to tell whether a notice is old. " +
    "If notices conflict or one revises another, prefer the one with the latest date. " +
    "Mention dates, deadlines, and programme names exactly as written. " +
    "If the excerpts do not contain the answer, say so and point to the most relevant notice. " +
    "Be brief: at most 5 sentences. " +
    `Today's date is ${today}.`;

  return [
    { role: "system", content: system },
    { role: "user", content: `Notice excerpts:\n\n${context}\n\nQuestion: ${question}` },
  ];
}

/** Retrieve, emit sources, then stream the model's answer. */
export async function* streamAnswer(question: string, signal?: AbortSignal): AsyncGenerator<RagEvent> {
  const chunks = await retrieveChunks(question, signal);
  if (chunks.length === 0) {
    yield { type: "sources", sources: [] };
    yield { type: "token", text: NOT_FOUND_ANSWER };
    yield { type: "done" };
    return;
  }

  const groups = groupByNotice(chunks);
  yield { type: "sources", sources: groups.map((g) => g.source) };

  const res = await postOllama(
    "/api/chat",
    {
      model: CHAT_MODEL,
      messages: buildMessages(question, groups),
      stream: true,
      keep_alive: KEEP_ALIVE,
      options: CHAT_OPTIONS,
    },
    CHAT_TIMEOUT_MS,
    signal,
  );
  if (!res.body) throw new RagError("Ollama returned an empty response.", 502);

  try {
    for await (const line of readLines(res.body)) {
      const data = JSON.parse(line) as { message?: { content?: string }; error?: string; done?: boolean };
      if (data.error) throw new RagError(`The language model failed: ${data.error}`, 502);
      if (data.message?.content) yield { type: "token", text: data.message.content };
      if (data.done) break;
    }
  } catch (err) {
    throw ollamaError(err, CHAT_TIMEOUT_MS, signal);
  }
  yield { type: "done" };
}

/** Non-streaming convenience wrapper: the whole answer plus sources. */
export async function answerQuestion(question: string, signal?: AbortSignal): Promise<RagResult> {
  let answer = "";
  let sources: Source[] = [];
  for await (const event of streamAnswer(question, signal)) {
    if (event.type === "sources") sources = event.sources;
    else if (event.type === "token") answer += event.text;
  }
  answer = answer.trim();
  if (!answer) throw new RagError("The language model returned an empty answer.", 502);
  return { answer, sources };
}
