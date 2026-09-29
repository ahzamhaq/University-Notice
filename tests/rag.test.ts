import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const rpc = vi.fn();
const createClient = vi.fn(() => ({ rpc }));
vi.mock("@supabase/supabase-js", () => ({ createClient }));

const EMBEDDING = new Array(1024).fill(0.01);

function chunk(url: string, date: string | null, similarity: number, content = "text") {
  return {
    id: 1,
    notice_id: 1,
    content,
    similarity,
    score: similarity,
    title: `Title ${url}`,
    url,
    notice_date: date,
    update_date: "",
  };
}

/** A streaming body that delivers `text` in pieces of `size` bytes (splitting lines mid-way). */
function streamOf(text: string, size = 7): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return new ReadableStream({
    pull(controller) {
      if (offset >= bytes.length) return controller.close();
      controller.enqueue(bytes.slice(offset, offset + size));
      offset += size;
    },
  });
}

function ollamaStream(tokens: string[]): string {
  return (
    tokens.map((t) => JSON.stringify({ message: { content: t }, done: false })).join("\n") +
    "\n" +
    JSON.stringify({ message: { content: "" }, done: true }) +
    "\n"
  );
}

/** Fake Ollama: records calls, answers /api/embed and streams /api/chat. */
function mockOllama(tokens = ["LLB exams ", "start 5 Jan ", "[1]."]) {
  const calls: { url: string; body: any }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: RequestInit) => {
      calls.push({ url, body: JSON.parse(String(init.body)) });
      if (url.endsWith("/api/embed")) return new Response(JSON.stringify({ embeddings: [EMBEDDING] }));
      return new Response(streamOf(ollamaStream(tokens)));
    }),
  );
  return calls;
}

async function loadRag() {
  vi.resetModules();
  return import("@/lib/rag");
}

async function collect<T>(gen: AsyncGenerator<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of gen) out.push(item);
  return out;
}

describe("lib/rag", () => {
  beforeEach(() => {
    process.env.SUPABASE_URL = "https://abc.supabase.co/rest/v1/";
    process.env.SUPABASE_KEY = "service-key";
    rpc.mockReset();
    createClient.mockClear();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("embeds the question as-is with local bge-m3 and keeps the model warm", async () => {
    const calls = mockOllama();
    const { embedQuery } = await loadRag();
    expect(await embedQuery("LLB datesheet")).toHaveLength(1024);
    expect(calls[0].url).toBe("http://localhost:11434/api/embed");
    expect(calls[0].body).toEqual({ model: "bge-m3", input: "LLB datesheet", keep_alive: "30m" });
  });

  it("names the real problem when the embedding model has the wrong size (e.g. old nomic model)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ embeddings: [new Array(768).fill(0.1)] }))),
    );
    const { embedQuery } = await loadRag();
    await expect(embedQuery("x")).rejects.toThrow(
      "The embedding model returned 768 dimensions but the database stores 1024. Check OLLAMA_EMBED_MODEL",
    );
  });

  it("strips /rest/v1/ from SUPABASE_URL and asks for 20 notices, 1 excerpt each, tuned recency", async () => {
    mockOllama();
    rpc.mockResolvedValue({ data: [], error: null });
    const { retrieveChunks } = await loadRag();
    await retrieveChunks("anything");
    expect(createClient).toHaveBeenCalledWith("https://abc.supabase.co", "service-key", expect.anything());
    expect(rpc).toHaveBeenCalledWith("match_notice_chunks", {
      query_embedding: EMBEDDING,
      match_count: 20,
      match_threshold: 0.45,
      max_per_notice: 1,
      recency_weight: 0.065,
    });
  });

  it("gives the model only the best 5 notices but returns all matches with citation numbers", async () => {
    const calls = mockOllama();
    const many = Array.from({ length: 8 }, (_, i) => chunk(`n${i + 1}.pdf`, "2026-09-01", 0.9 - i * 0.01));
    rpc.mockResolvedValue({ data: many, error: null });
    const { answerQuestion } = await loadRag();
    const { sources } = await answerQuestion("q?");
    expect(sources.map((s) => s.ref)).toEqual([1, 2, 3, 4, 5, null, null, null]);
    const prompt = calls.find((c) => c.url.endsWith("/api/chat"))!.body.messages[1].content as string;
    expect(prompt).toContain("[5] TITLE: Title n5.pdf");
    expect(prompt).not.toContain("n6.pdf"); // extra matches are shown in the UI, not sent to the model
  });

  it("answers 'not found' without calling the LLM when nothing matches", async () => {
    const calls = mockOllama();
    rpc.mockResolvedValue({ data: [], error: null });
    const { answerQuestion, NOT_FOUND_ANSWER } = await loadRag();
    expect(await answerQuestion("unrelated question")).toEqual({ answer: NOT_FOUND_ANSWER, sources: [] });
    expect(calls.some((c) => c.url.endsWith("/api/chat"))).toBe(false);
  });

  it("emits sources before the model is even called", async () => {
    const calls = mockOllama();
    rpc.mockResolvedValue({ data: [chunk("a.pdf", "2026-09-01", 0.8)], error: null });
    const { streamAnswer } = await loadRag();
    const gen = streamAnswer("q");
    const first = await gen.next();
    expect(first.value).toMatchObject({ type: "sources" });
    expect(calls.some((c) => c.url.endsWith("/api/chat"))).toBe(false);
    await collect(gen);
  });

  it("streams tokens in order even when NDJSON lines are split across network chunks", async () => {
    mockOllama(["Theory ", "exams ", "start ", "5 Jan [1]."]);
    rpc.mockResolvedValue({ data: [chunk("a.pdf", "2026-09-01", 0.8)], error: null });
    const { streamAnswer } = await loadRag();
    const events = await collect(streamAnswer("q"));
    expect(events.map((e) => e.type)).toEqual(["sources", "token", "token", "token", "token", "done"]);
    expect(events.filter((e) => e.type === "token").map((e: any) => e.text).join("")).toBe(
      "Theory exams start 5 Jan [1].",
    );
  });

  it("keeps the database's score order so [1] is the best match, even if older", async () => {
    const calls = mockOllama();
    rpc.mockResolvedValue({
      data: [
        chunk("best-old.pdf", "2025-09-10", 0.79),
        chunk("new.pdf", "2026-09-25", 0.67),
        chunk("undated.pdf", null, 0.6),
      ],
      error: null,
    });
    const { answerQuestion } = await loadRag();
    const result = await answerQuestion("kabaddi trials");
    expect(result.sources.map((s) => s.url)).toEqual(["best-old.pdf", "new.pdf", "undated.pdf"]);
    const prompt = calls.find((c) => c.url.endsWith("/api/chat"))!.body.messages[1].content as string;
    expect(prompt.indexOf("[1] TITLE: Title best-old.pdf")).toBeGreaterThan(-1);
    expect(prompt.indexOf("[2] TITLE: Title new.pdf")).toBeGreaterThan(prompt.indexOf("[1] TITLE: Title best-old.pdf"));
  });

  it("keeps the prompt small: trimmed excerpts, 4k context, capped answer length", async () => {
    const calls = mockOllama();
    rpc.mockResolvedValue({ data: [chunk("a.pdf", "2026-09-01", 0.8, "word ".repeat(1000))], error: null });
    const { answerQuestion } = await loadRag();
    await answerQuestion("q");
    const chat = calls.find((c) => c.url.endsWith("/api/chat"))!.body;
    expect(chat.messages[1].content.length).toBeLessThan(1300); // 5000-char excerpt trimmed to ~1000
    expect(chat.options).toMatchObject({ num_ctx: 4096, num_predict: 350 });
    expect(chat.stream).toBe(true);
    expect(chat.keep_alive).toBe("30m");
    expect(chat.messages[0].content).toMatch(/most relevant first/);
    expect(chat.messages[0].content).toMatch(/prefer the one with the latest date/);
    expect(chat.messages[0].content).toMatch(/NEVER attach a date, time, venue or detail from one notice to another/);
  });

  it("marks scanned (title-only) notices as having no details, so the model can't borrow them", async () => {
    const calls = mockOllama();
    const titleOnly =
      "Notice: Kabaddi trials (dated 2025-09-10)\n\n(Only the title of this notice is available: " +
      "the PDF is a scanned image. Open the PDF for the full details.)";
    rpc.mockResolvedValue({
      data: [
        chunk("scan.pdf", "2025-09-10", 0.8, titleOnly),
        chunk("text.pdf", "2026-09-25", 0.7, "Trials on 30 Sep at 15:30"),
      ],
      error: null,
    });
    const { answerQuestion } = await loadRag();
    await answerQuestion("kabaddi trials");
    const prompt = calls.find((c) => c.url.endsWith("/api/chat"))!.body.messages[1].content as string;
    expect(prompt).toContain(
      "[1] TITLE: Title scan.pdf\nNOTICE DATE: 2025-09-10\nCONTENT: not available (scanned PDF)",
    );
    expect(prompt).toContain("it has NO dates, times or venues");
    expect(prompt).toContain("[2] TITLE: Title text.pdf\nNOTICE DATE: 2026-09-25\nCONTENT:\nTrials on 30 Sep at 15:30");
  });

  it("trimExcerpt cuts at a word boundary and leaves short text alone", async () => {
    const { trimExcerpt } = await loadRag();
    expect(trimExcerpt("short text", 100)).toBe("short text");
    const trimmed = trimExcerpt("alpha beta gamma delta", 13);
    expect(trimmed).toBe("alpha beta …");
  });

  it("turns an error line in the model stream into a RagError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.endsWith("/api/embed")
          ? new Response(JSON.stringify({ embeddings: [EMBEDDING] }))
          : new Response(streamOf(JSON.stringify({ error: "out of memory" }) + "\n")),
      ),
    );
    rpc.mockResolvedValue({ data: [chunk("a.pdf", "2026-09-01", 0.8)], error: null });
    const { answerQuestion } = await loadRag();
    await expect(answerQuestion("q")).rejects.toThrow(/out of memory/);
  });

  it("turns an unreachable Ollama into a 503 RagError", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("fetch failed"))));
    const { embedQuery, RagError } = await loadRag();
    const err = await embedQuery("x").catch((e) => e);
    expect(err).toBeInstanceOf(RagError);
    expect(err.status).toBe(503);
  });

  it("reports a slow model as a 504 timeout, not as Ollama being unreachable", async () => {
    const timeout = new DOMException("The operation was aborted due to timeout", "TimeoutError");
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(timeout)));
    const { embedQuery } = await loadRag();
    const err = await embedQuery("x").catch((e) => e);
    expect(err.status).toBe(504);
    expect(err.message).toMatch(/took longer than/);
  });

  it("passes a client disconnect through untouched (not reported as an Ollama error)", async () => {
    const controller = new AbortController();
    controller.abort();
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new DOMException("aborted", "AbortError"))));
    const { embedQuery, RagError } = await loadRag();
    const err = await embedQuery("x", controller.signal).catch((e) => e);
    expect(err).not.toBeInstanceOf(RagError);
    expect(err.name).toBe("AbortError");
  });

  it("turns a missing model (Ollama 404) into a 502 with a hint", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("model not found", { status: 404 })));
    const { embedQuery } = await loadRag();
    const err = await embedQuery("x").catch((e) => e);
    expect(err.status).toBe(502);
    expect(err.message).toContain("bge-m3");
  });

  it("reports a vector search failure as a RagError", async () => {
    mockOllama();
    rpc.mockResolvedValue({ data: null, error: { message: "function does not exist" } });
    const { retrieveChunks } = await loadRag();
    await expect(retrieveChunks("x")).rejects.toThrow(/schema.sql/);
  });

  it("fails clearly when Supabase env vars are missing", async () => {
    mockOllama();
    delete process.env.SUPABASE_URL;
    const { retrieveChunks } = await loadRag();
    await expect(retrieveChunks("x")).rejects.toThrow(/not configured/);
  });
});
