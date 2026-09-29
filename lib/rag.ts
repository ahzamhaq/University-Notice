import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// Server-only RAG pipeline: embed the question, find the best-matching notices in Supabase,
// then stream an answer from a chat model.
//
// Two interchangeable backends, chosen by env vars:
// - Local (default, works offline): Ollama for both embeddings and chat.
// - Hosted (for serverless deploys like Vercel, where Ollama can't run): Cloudflare Workers AI.
//   EMBED_PROVIDER=cloudflare runs the same bge-m3 model that ingestion stores (identical
//   vectors), and LLM_PROVIDER=cloudflare streams answers from a hosted Llama model.
//
// Sources are emitted as soon as retrieval finishes and the answer is streamed token by token.

const OLLAMA_ENDPOINT = (process.env.OLLAMA_ENDPOINT ?? "http://localhost:11434").replace(/\/+$/, "");
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL ?? "bge-m3";
const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL ?? "llama3.2";

const EMBED_PROVIDER = (process.env.EMBED_PROVIDER ?? "ollama").trim().toLowerCase();
const LLM_PROVIDER = (process.env.LLM_PROVIDER ?? "ollama").trim().toLowerCase();
const CF_EMBED_MODEL = "@cf/baai/bge-m3";
const EMBED_DIM = 1024; // vector(1024) in supabase/schema.sql; config.EMBED_DIM on the Python side
const CF_CHAT_MODEL = process.env.CLOUDFLARE_CHAT_MODEL ?? "@cf/meta/llama-3.1-8b-instruct-fp8-fast";

const SOURCE_COUNT = 20; // matching notices returned to the UI ("show more")
const CONTEXT_COUNT = 5; // best of those given to the model; keeps answers fast and cheap
// bge-m3 scores relevant notices ~0.5-0.7 and unrelated text ~0.3-0.41, so 0.45 filters junk.
const MATCH_THRESHOLD = 0.45;
// score = similarity + RECENCY_WEIGHT * 0.5^(age_days / 180). Tuned for bge-m3: 0.065 puts this
// year's notices above near-identical old ones; 0.07+ let new but less relevant notices (e.g.
// NSS for an "NSP scholarship" question) push the right answer down.
const RECENCY_WEIGHT = 0.065;
const MAX_CHUNKS_PER_NOTICE = 1;
const MAX_EXCERPT_CHARS = 1000; // ~250 tokens per notice keeps the prompt near 1.5k tokens
const EMBED_TIMEOUT_MS = 60_000;
// CPU-only Ollama can take minutes; hosted APIs answer in seconds and serverless functions
// have short limits, so fail fast there.
const CHAT_TIMEOUT_MS = LLM_PROVIDER === "cloudflare" ? 55_000 : 300_000;
// Keep Ollama models loaded between questions so only the first one pays the load time.
const KEEP_ALIVE = "30m";
const TEMPERATURE = 0.2;
const MAX_ANSWER_TOKENS = 350;
const OLLAMA_OPTIONS = { temperature: TEMPERATURE, num_ctx: 4096, num_predict: MAX_ANSWER_TOKENS };

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
  /** Citation number used in the answer ([1]..[5]); null for extra notices the model didn't read. */
  ref: number | null;
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

type ChatMessage = { role: "system" | "user"; content: string };

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

function requireEnv(name: "CLOUDFLARE_ACCOUNT_ID" | "CLOUDFLARE_API_TOKEN"): string {
  const value = process.env[name]?.trim();
  if (!value) throw new RagError(`${name} is not set on the server.`);
  return value;
}

const CLOUDFLARE: Service = { name: "Cloudflare Workers AI", local: false, model: "" };

/** URL and auth header for a Workers AI model (read at call time so tests can change env). */
function cloudflare(model: string): { url: string; headers: Record<string, string> } {
  const account = requireEnv("CLOUDFLARE_ACCOUNT_ID");
  const token = requireEnv("CLOUDFLARE_API_TOKEN");
  return {
    url: `https://api.cloudflare.com/client/v4/accounts/${account}/ai/run/${model}`,
    headers: { Authorization: `Bearer ${token}` },
  };
}

// --- HTTP helpers -----------------------------------------------------------------------

interface Service {
  name: string; // shown to users, e.g. "Ollama", "Cloudflare Workers AI"
  local: boolean;
  model: string;
}

const OLLAMA: Service = { name: "Ollama", local: true, model: "" };

function withTimeout(timeoutMs: number, signal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([timeout, signal]) : timeout;
}

/** Map fetch/stream failures to user-facing errors. Client disconnects are rethrown as-is. */
function networkError(err: unknown, service: Service, timeoutMs: number, signal?: AbortSignal): unknown {
  if (signal?.aborted || err instanceof RagError) return err;
  if (err instanceof Error && err.name === "TimeoutError") {
    const what = service.local ? "The local model" : `The ${service.name} service`;
    return new RagError(
      `${what} took longer than ${timeoutMs / 1000}s to respond. It may still be loading; try again.`,
      504,
      { cause: err },
    );
  }
  const message = service.local
    ? `Could not reach Ollama at ${OLLAMA_ENDPOINT}. Is \`ollama serve\` running?`
    : `Could not reach the ${service.name} API.`;
  return new RagError(message, 503, { cause: err });
}

function httpError(status: number, detail: string, service: Service): RagError {
  if (service.local) {
    return new RagError(`Ollama returned ${status}. Has the model "${service.model}" been pulled?`, 502, {
      cause: detail,
    });
  }
  if (status === 401 || status === 403) {
    return new RagError(`The ${service.name} API key is missing or invalid.`, 502, { cause: detail });
  }
  if (status === 429) {
    return new RagError(`The ${service.name} service is busy (rate limit). Try again in a minute.`, 503, {
      cause: detail,
    });
  }
  return new RagError(`The ${service.name} API returned ${status}.`, 502, { cause: detail });
}

async function postJson(
  url: string,
  body: object,
  service: Service,
  timeoutMs: number,
  signal?: AbortSignal,
  headers: Record<string, string> = {},
): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
      signal: withTimeout(timeoutMs, signal),
    });
  } catch (err) {
    throw networkError(err, service, timeoutMs, signal);
  }
  if (!res.ok) throw httpError(res.status, await res.text().catch(() => ""), service);
  return res;
}

/** Yield complete lines from a streaming response body (NDJSON or server-sent events). */
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

// --- Embeddings ---------------------------------------------------------------------------

export async function embedQuery(question: string, signal?: AbortSignal): Promise<number[]> {
  // bge-m3 uses no task prefix: questions and documents are embedded as-is.
  let vector: number[] | undefined;
  if (EMBED_PROVIDER === "cloudflare") {
    const { url, headers } = cloudflare(CF_EMBED_MODEL);
    const res = await postJson(
      url,
      { text: [question] },
      { ...CLOUDFLARE, model: CF_EMBED_MODEL },
      EMBED_TIMEOUT_MS,
      signal,
      headers,
    );
    vector = ((await res.json()) as { result?: { data?: number[][] } }).result?.data?.[0];
  } else {
    const res = await postJson(
      `${OLLAMA_ENDPOINT}/api/embed`,
      { model: EMBED_MODEL, input: question, keep_alive: KEEP_ALIVE },
      { ...OLLAMA, model: EMBED_MODEL },
      EMBED_TIMEOUT_MS,
      signal,
    );
    vector = ((await res.json()) as { embeddings?: number[][] }).embeddings?.[0];
  }
  if (!vector?.length) throw new RagError("The embedding service returned an empty embedding.", 502);
  if (vector.length !== EMBED_DIM) {
    // e.g. a leftover OLLAMA_EMBED_MODEL=nomic-embed-text (768) against the bge-m3 (1024) database.
    throw new RagError(
      `The embedding model returned ${vector.length} dimensions but the database stores ${EMBED_DIM}. ` +
        "Check OLLAMA_EMBED_MODEL (it should be bge-m3).",
      500,
    );
  }
  return vector;
}

export async function retrieveChunks(question: string, signal?: AbortSignal): Promise<NoticeChunk[]> {
  const embedding = await embedQuery(question, signal);
  const { data, error } = await getSupabase().rpc("match_notice_chunks", {
    query_embedding: embedding,
    match_count: SOURCE_COUNT,
    match_threshold: MATCH_THRESHOLD,
    max_per_notice: MAX_CHUNKS_PER_NOTICE,
    recency_weight: RECENCY_WEIGHT,
  });
  if (error) {
    throw new RagError("Vector search failed. Has supabase/schema.sql been applied?", 502, { cause: error });
  }
  return (data ?? []) as NoticeChunk[];
}

// --- Prompt ---------------------------------------------------------------------------------

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
          ref: null,
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

function buildMessages(question: string, groups: NoticeGroup[]): ChatMessage[] {
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

// --- Chat (streaming) -------------------------------------------------------------------------

/** Ollama streams NDJSON: {"message":{"content":"..."},"done":false} per line. */
async function* streamOllamaChat(messages: ChatMessage[], signal?: AbortSignal): AsyncGenerator<string> {
  const service = { ...OLLAMA, model: CHAT_MODEL };
  const res = await postJson(
    `${OLLAMA_ENDPOINT}/api/chat`,
    { model: CHAT_MODEL, messages, stream: true, keep_alive: KEEP_ALIVE, options: OLLAMA_OPTIONS },
    service,
    CHAT_TIMEOUT_MS,
    signal,
  );
  if (!res.body) throw new RagError("Ollama returned an empty response.", 502);
  try {
    for await (const line of readLines(res.body)) {
      const data = JSON.parse(line) as { message?: { content?: string }; error?: string; done?: boolean };
      if (data.error) throw new RagError(`The language model failed: ${data.error}`, 502);
      if (data.message?.content) yield data.message.content;
      if (data.done) break;
    }
  } catch (err) {
    throw networkError(err, service, CHAT_TIMEOUT_MS, signal);
  }
}

/**
 * Workers AI streams server-sent events. Current models send OpenAI-style
 * `data: {"choices":[{"delta":{"content":"..."}}]}`; older ones send `data: {"response":"..."}`.
 */
async function* streamCloudflareChat(messages: ChatMessage[], signal?: AbortSignal): AsyncGenerator<string> {
  const service = { ...CLOUDFLARE, model: CF_CHAT_MODEL };
  const { url, headers } = cloudflare(CF_CHAT_MODEL);
  const res = await postJson(
    url,
    { messages, stream: true, temperature: TEMPERATURE, max_tokens: MAX_ANSWER_TOKENS },
    service,
    CHAT_TIMEOUT_MS,
    signal,
    headers,
  );
  if (!res.body) throw new RagError("Cloudflare returned an empty response.", 502);
  try {
    for await (const line of readLines(res.body)) {
      if (!line.startsWith("data:")) continue; // comments / keep-alives
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") break;
      const data = JSON.parse(payload) as {
        choices?: { delta?: { content?: string } }[];
        response?: string;
        error?: { message?: string } | string;
      };
      if (data.error) {
        const detail = typeof data.error === "string" ? data.error : (data.error.message ?? "unknown error");
        throw new RagError(`The language model failed: ${detail}`, 502);
      }
      const text = data.choices?.[0]?.delta?.content ?? data.response;
      if (text) yield text;
    }
  } catch (err) {
    throw networkError(err, service, CHAT_TIMEOUT_MS, signal);
  }
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
  // The model reads the best CONTEXT_COUNT notices (cited as [1]..[n]); the rest are extra
  // matches the UI can reveal with "show more".
  const context = groups.slice(0, CONTEXT_COUNT);
  yield {
    type: "sources",
    sources: groups.map((g, i) => ({ ...g.source, ref: i < CONTEXT_COUNT ? i + 1 : null })),
  };

  const messages = buildMessages(question, context);
  const chat =
    LLM_PROVIDER === "cloudflare" ? streamCloudflareChat(messages, signal) : streamOllamaChat(messages, signal);
  for await (const text of chat) yield { type: "token", text };
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
