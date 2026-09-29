import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Hosted mode used on Vercel: EMBED_PROVIDER=cloudflare, LLM_PROVIDER=cloudflare.
const rpc = vi.fn();
vi.mock("@supabase/supabase-js", () => ({ createClient: vi.fn(() => ({ rpc })) }));

const EMBEDDING = new Array(1024).fill(0.02);
const CF = "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run";
const CHUNK = {
  id: 1,
  notice_id: 1,
  content: "Notice: LLB datesheet\n\nTheory exams start 5 Jan 2027.",
  similarity: 0.8,
  score: 0.85,
  title: "LLB datesheet",
  url: "https://www.ipu.ac.in/a.pdf",
  notice_date: "2026-09-28",
  update_date: "",
};

function streamOf(text: string, size = 9): ReadableStream<Uint8Array> {
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

/** OpenAI-style SSE, as Workers AI Llama 3.1 streams it (captured from the real API). */
function sse(tokens: string[]): string {
  const first = `data: ${JSON.stringify({ choices: [{ delta: { content: "", role: "assistant" } }] })}\n\n`;
  const events = tokens.map((t) => `data: ${JSON.stringify({ choices: [{ delta: { content: t } }] })}\n\n`);
  const stop = `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`;
  return first + events.join("") + stop + "data: [DONE]\n\n";
}

type Call = { url: string; body: any; headers: Record<string, string> };

function mockCloudflare(chat: () => Response = () => new Response(streamOf(sse(["Exams ", "start ", "5 Jan [1]."])))) {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: RequestInit) => {
      calls.push({ url, body: JSON.parse(String(init.body)), headers: init.headers as Record<string, string> });
      if (url.endsWith("/@cf/baai/bge-m3")) {
        return new Response(JSON.stringify({ success: true, result: { shape: [1, 1024], data: [EMBEDDING] } }));
      }
      return chat();
    }),
  );
  return calls;
}

async function loadRag() {
  vi.resetModules();
  return import("@/lib/rag");
}

const HOSTED_ENV = {
  SUPABASE_URL: "https://abc.supabase.co",
  SUPABASE_KEY: "service-key",
  EMBED_PROVIDER: "cloudflare",
  LLM_PROVIDER: "cloudflare",
  CLOUDFLARE_ACCOUNT_ID: "acct123",
  CLOUDFLARE_API_TOKEN: "cf-token",
};

describe("lib/rag in hosted mode (Cloudflare Workers AI)", () => {
  beforeEach(() => {
    Object.assign(process.env, HOSTED_ENV);
    rpc.mockReset();
    rpc.mockResolvedValue({ data: [CHUNK], error: null });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    for (const key of ["EMBED_PROVIDER", "LLM_PROVIDER", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"]) {
      delete process.env[key];
    }
  });

  it("embeds the question with Cloudflare bge-m3 (no prefix) using the account and token", async () => {
    const calls = mockCloudflare();
    const { embedQuery } = await loadRag();
    expect(await embedQuery("LLB datesheet")).toEqual(EMBEDDING);
    expect(calls[0].url).toBe(`${CF}/@cf/baai/bge-m3`);
    expect(calls[0].body).toEqual({ text: ["LLB datesheet"] });
    expect(calls[0].headers.Authorization).toBe("Bearer cf-token");
  });

  it("streams the answer from Workers AI server-sent events split across network chunks", async () => {
    const calls = mockCloudflare();
    const { answerQuestion } = await loadRag();
    const result = await answerQuestion("When do LLB exams start?");
    expect(result.answer).toBe("Exams start 5 Jan [1].");
    expect(result.sources[0].url).toBe(CHUNK.url);
    const chat = calls.find((c) => c.url.includes("llama"))!;
    expect(chat.url).toBe(`${CF}/@cf/meta/llama-3.1-8b-instruct-fp8-fast`);
    expect(chat.headers.Authorization).toBe("Bearer cf-token");
    expect(chat.body).toMatchObject({ stream: true, max_tokens: 350 });
    expect(chat.body.messages[0].role).toBe("system");
  });

  it("also understands the older {response: ...} stream format", async () => {
    const legacy = ["Exams ", "start."].map((t) => `data: ${JSON.stringify({ response: t })}\n\n`).join("") + "data: [DONE]\n\n";
    mockCloudflare(() => new Response(streamOf(legacy)));
    const { answerQuestion } = await loadRag();
    expect((await answerQuestion("q?")).answer).toBe("Exams start.");
  });

  it("never calls Ollama in hosted mode", async () => {
    const calls = mockCloudflare();
    const { answerQuestion } = await loadRag();
    await answerQuestion("q?");
    expect(calls.some((c) => c.url.includes("11434"))).toBe(false);
  });

  it("explains missing Cloudflare settings instead of calling the service", async () => {
    delete process.env.CLOUDFLARE_API_TOKEN;
    const calls = mockCloudflare();
    const { embedQuery } = await loadRag();
    await expect(embedQuery("q?")).rejects.toThrow("CLOUDFLARE_API_TOKEN is not set on the server.");
    expect(calls).toHaveLength(0);
  });

  it("reports a rejected token clearly", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response('{"success":false,"errors":[{"code":10000}]}', { status: 401 })));
    const { embedQuery } = await loadRag();
    await expect(embedQuery("q?")).rejects.toThrow("The Cloudflare Workers AI API key is missing or invalid.");
  });

  it("turns the daily free limit (429) into a retryable 503", async () => {
    mockCloudflare(() => new Response("rate limited", { status: 429 }));
    const { answerQuestion } = await loadRag();
    const err = await answerQuestion("q?").catch((e) => e);
    expect(err.status).toBe(503);
    expect(err.message).toMatch(/busy \(rate limit\)/);
  });

  it("says the hosted API is unreachable, not Ollama", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("fetch failed"))));
    const { embedQuery } = await loadRag();
    const err = await embedQuery("x").catch((e) => e);
    expect(err.message).toBe("Could not reach the Cloudflare Workers AI API.");
  });

  it("surfaces an error event inside the stream", async () => {
    mockCloudflare(() => new Response(streamOf(`data: ${JSON.stringify({ error: "model overloaded" })}\n\n`)));
    const { answerQuestion } = await loadRag();
    await expect(answerQuestion("q?")).rejects.toThrow(/model overloaded/);
  });
});
