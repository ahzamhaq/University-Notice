import { NextRequest, NextResponse } from "next/server";

import { logger } from "@/lib/logger";
import { answerQuestion, RagError, streamAnswer } from "@/lib/rag";
import { checkRateLimit } from "@/lib/rate-limit";

export const runtime = "nodejs"; // logger needs fs
export const dynamic = "force-dynamic";
// Vercel function time limit (seconds). Hosted models answer well within this; local Ollama
// has no such limit when self-hosted.
export const maxDuration = 60;

const MIN_QUESTION_LENGTH = 3;
const MAX_QUESTION_LENGTH = 500;
const NDJSON = "application/x-ndjson";

function clientIp(req: NextRequest): string {
  const forwarded = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return forwarded || req.headers.get("x-real-ip") || "unknown";
}

function publicError(err: unknown): { status: number; message: string } {
  return err instanceof RagError
    ? { status: err.status, message: err.message }
    : { status: 500, message: "Something went wrong answering that question." };
}

/**
 * POST { question }.
 * - Default: JSON `{ answer, sources }` once the whole answer is ready.
 * - With `Accept: application/x-ndjson`: one JSON event per line, as soon as each is ready:
 *   `{type:"sources"}` (after ~3s), then `{type:"token"}`s, then `{type:"done"}` or `{type:"error"}`.
 */
export async function POST(req: NextRequest) {
  const ip = clientIp(req);
  const limit = checkRateLimit(ip);
  const headers: Record<string, string> = {
    "X-RateLimit-Limit": String(limit.limit),
    "X-RateLimit-Remaining": String(limit.remaining),
  };

  if (!limit.allowed) {
    logger.warn("Rate limit exceeded", { ip });
    return NextResponse.json(
      { error: `Too many requests. Try again in ${limit.retryAfterSeconds}s.` },
      { status: 429, headers: { ...headers, "Retry-After": String(limit.retryAfterSeconds) } },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Request body must be JSON." }, { status: 400, headers });
  }

  const raw = (body as { question?: unknown } | null)?.question;
  const question = typeof raw === "string" ? raw.trim() : "";
  if (question.length < MIN_QUESTION_LENGTH || question.length > MAX_QUESTION_LENGTH) {
    return NextResponse.json(
      { error: `"question" must be between ${MIN_QUESTION_LENGTH} and ${MAX_QUESTION_LENGTH} characters.` },
      { status: 400, headers },
    );
  }

  if (req.headers.get("accept")?.includes(NDJSON)) {
    return streamResponse(question, ip, req.signal, headers);
  }

  const started = Date.now();
  try {
    const result = await answerQuestion(question, req.signal);
    logger.info("Answered query", { ip, ms: Date.now() - started, sources: result.sources.length });
    return NextResponse.json(result, { headers });
  } catch (err) {
    logger.error("Query failed", err, { ip, question });
    const { status, message } = publicError(err);
    return NextResponse.json({ error: message }, { status, headers });
  }
}

function streamResponse(question: string, ip: string, signal: AbortSignal, headers: Record<string, string>) {
  const encoder = new TextEncoder();
  const started = Date.now();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: object) => {
        try {
          controller.enqueue(encoder.encode(JSON.stringify(event) + "\n"));
        } catch {
          // Client already gone; nothing to deliver.
        }
      };
      try {
        for await (const event of streamAnswer(question, signal)) send(event);
        logger.info("Streamed answer", { ip, ms: Date.now() - started });
      } catch (err) {
        if (signal.aborted) {
          // The user closed the page or asked something else; generation was cancelled.
          logger.info("Client disconnected; generation cancelled", { ip, ms: Date.now() - started });
        } else {
          logger.error("Query failed", err, { ip, question });
          send({ type: "error", error: publicError(err).message });
        }
      } finally {
        try {
          controller.close();
        } catch {
          // Already closed by a disconnect.
        }
      }
    },
  });

  return new Response(stream, {
    headers: { ...headers, "Content-Type": `${NDJSON}; charset=utf-8`, "Cache-Control": "no-cache, no-transform" },
  });
}
