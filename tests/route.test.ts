import { NextRequest } from "next/server";
import { describe, expect, it, vi } from "vitest";

const answerQuestion = vi.fn();
const streamAnswer = vi.fn();

vi.mock("@/lib/logger", () => ({ logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() } }));
vi.mock("@/lib/rag", async () => {
  const actual = await vi.importActual<typeof import("@/lib/rag")>("@/lib/rag");
  return { RagError: actual.RagError, answerQuestion, streamAnswer };
});

const { POST } = await import("@/app/api/query/route");
const { RagError } = await import("@/lib/rag");

let ipCounter = 0;
function request(body: string, ip = `10.0.0.${++ipCounter}`, accept = "application/json") {
  return new NextRequest("http://localhost/api/query", {
    method: "POST",
    headers: { "content-type": "application/json", "x-forwarded-for": `${ip}, 172.16.0.1`, accept },
    body,
  });
}

describe("POST /api/query", () => {
  it("returns the answer and sources with rate-limit headers", async () => {
    answerQuestion.mockResolvedValue({ answer: "Yes [1]", sources: [{ title: "LLB", url: "u" }] });
    const res = await POST(request(JSON.stringify({ question: "  LLB datesheet?  " })));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ answer: "Yes [1]", sources: [{ title: "LLB", url: "u" }] });
    expect(answerQuestion.mock.calls.at(-1)?.[0]).toBe("LLB datesheet?"); // trimmed
    expect(res.headers.get("x-ratelimit-limit")).toBe("50");
    expect(res.headers.get("x-ratelimit-remaining")).toBe("49");
  });

  it.each([
    ["invalid JSON", "not json", /must be JSON/],
    ["missing question", "{}", /between 3 and 500/],
    ["non-string question", JSON.stringify({ question: 42 }), /between 3 and 500/],
    ["too short", JSON.stringify({ question: " a " }), /between 3 and 500/],
    ["too long", JSON.stringify({ question: "x".repeat(501) }), /between 3 and 500/],
    ["null body", "null", /between 3 and 500/],
  ])("rejects %s with 400", async (_name, body, message) => {
    const callsBefore = answerQuestion.mock.calls.length;
    const res = await POST(request(body));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toMatch(message);
    expect(answerQuestion.mock.calls.length).toBe(callsBefore);
  });

  it("passes RagError status and message through (e.g. Ollama down)", async () => {
    answerQuestion.mockImplementation(async () => {
      throw new RagError("Could not reach Ollama", 503);
    });
    const res = await POST(request(JSON.stringify({ question: "exam dates" })));
    expect(res.status).toBe(503);
    expect((await res.json()).error).toBe("Could not reach Ollama");
  });

  it("hides unexpected error details behind a generic 500", async () => {
    answerQuestion.mockImplementation(async () => {
      throw new Error("secret db password in stack");
    });
    const res = await POST(request(JSON.stringify({ question: "exam dates" })));
    expect(res.status).toBe(500);
    expect(JSON.stringify(await res.json())).not.toContain("secret");
  });

  it("returns 429 with Retry-After after 50 requests from one IP", async () => {
    answerQuestion.mockResolvedValue({ answer: "ok", sources: [] });
    const body = JSON.stringify({ question: "exam dates" });
    for (let i = 0; i < 50; i++) expect((await POST(request(body, "203.0.113.9"))).status).toBe(200);
    const res = await POST(request(body, "203.0.113.9"));
    expect(res.status).toBe(429);
    expect(Number(res.headers.get("retry-after"))).toBeGreaterThan(0);
    expect((await POST(request(body, "203.0.113.10"))).status).toBe(200); // other IPs unaffected
  });

  describe("streaming (Accept: application/x-ndjson)", () => {
    const NDJSON = "application/x-ndjson";

    async function events(res: Response) {
      return (await res.text())
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
    }

    it("streams sources, then tokens, then done, as NDJSON", async () => {
      streamAnswer.mockImplementation(async function* () {
        yield { type: "sources", sources: [{ title: "LLB", url: "u", noticeDate: "2026-09-28", similarity: 0.8 }] };
        yield { type: "token", text: "Yes " };
        yield { type: "token", text: "[1]" };
        yield { type: "done" };
      });
      const res = await POST(request(JSON.stringify({ question: "LLB datesheet?" }), undefined, NDJSON));
      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toContain(NDJSON);
      expect(res.headers.get("x-ratelimit-limit")).toBe("50");
      expect((await events(res)).map((e) => e.type)).toEqual(["sources", "token", "token", "done"]);
      expect(streamAnswer.mock.calls.at(-1)?.[1]).toBeInstanceOf(AbortSignal); // cancellable
    });

    it("sends a readable error event if the model fails mid-answer", async () => {
      streamAnswer.mockImplementation(async function* () {
        yield { type: "sources", sources: [] };
        throw new RagError("The local model took longer than 300s to respond.", 504);
      });
      const res = await POST(request(JSON.stringify({ question: "exam dates" }), undefined, NDJSON));
      const evts = await events(res);
      expect(evts.map((e) => e.type)).toEqual(["sources", "error"]);
      expect(evts[1].error).toMatch(/took longer than/);
    });

    it("hides unexpected error details in the stream too", async () => {
      streamAnswer.mockImplementation(async function* () {
        throw new Error("secret db password in stack");
      });
      const res = await POST(request(JSON.stringify({ question: "exam dates" }), undefined, NDJSON));
      const evts = await events(res);
      expect(evts).toEqual([{ type: "error", error: "Something went wrong answering that question." }]);
    });

    it("still validates input with a normal 400 JSON response", async () => {
      const res = await POST(request(JSON.stringify({ question: "a" }), undefined, NDJSON));
      expect(res.status).toBe(400);
      expect((await res.json()).error).toMatch(/between 3 and 500/);
    });
  });
});
