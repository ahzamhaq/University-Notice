import { appendFile, mkdir } from "node:fs/promises";
import path from "node:path";

// Server-only: writes JSON lines to the console and to logs/api.log.
// On Vercel the filesystem is read-only, so only the console is used (Vercel keeps those logs).
const LOG_DIR = path.join(process.cwd(), "logs");
const LOG_FILE = path.join(LOG_DIR, "api.log");
const WRITE_FILE = !process.env.VERCEL;

type Level = "info" | "warn" | "error";
type Meta = Record<string, unknown>;

let dirReady: Promise<unknown> | null = null;

function describeError(err: unknown): Meta {
  if (err instanceof Error) {
    const cause = err.cause instanceof Error ? err.cause.message : err.cause;
    return { error: err.message, stack: err.stack, ...(cause !== undefined && { cause }) };
  }
  return { error: String(err) };
}

function write(level: Level, message: string, meta: Meta = {}): void {
  const line = JSON.stringify({ time: new Date().toISOString(), level, message, ...meta });
  if (level === "error") console.error(line);
  else if (level === "warn") console.warn(line);
  else console.log(line);

  if (!WRITE_FILE) return;
  dirReady ??= mkdir(LOG_DIR, { recursive: true });
  dirReady
    .then(() => appendFile(LOG_FILE, line + "\n", "utf8"))
    .catch((err) => console.error("Failed to write logs/api.log:", err));
}

export const logger = {
  info: (message: string, meta?: Meta) => write("info", message, meta),
  warn: (message: string, meta?: Meta) => write("warn", message, meta),
  error: (message: string, err?: unknown, meta?: Meta) =>
    write("error", message, { ...meta, ...(err !== undefined && describeError(err)) }),
};
