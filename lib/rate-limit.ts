// Sliding-window rate limiter kept in memory. Limits are per server process, which is
// fine for a single `next start` instance; use Redis/Upstash if you run several.
const WINDOW_MS = 60_000;
const MAX_REQUESTS = 50;

const hits = new Map<string, number[]>();

export interface RateLimitResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  retryAfterSeconds: number;
}

export function checkRateLimit(key: string, now = Date.now()): RateLimitResult {
  const windowStart = now - WINDOW_MS;
  const recent = (hits.get(key) ?? []).filter((t) => t > windowStart);

  if (recent.length >= MAX_REQUESTS) {
    hits.set(key, recent);
    const retryAfterSeconds = Math.max(1, Math.ceil((recent[0] + WINDOW_MS - now) / 1000));
    return { allowed: false, limit: MAX_REQUESTS, remaining: 0, retryAfterSeconds };
  }

  recent.push(now);
  hits.set(key, recent);
  if (hits.size > 10_000) sweep(windowStart);
  return { allowed: true, limit: MAX_REQUESTS, remaining: MAX_REQUESTS - recent.length, retryAfterSeconds: 0 };
}

function sweep(windowStart: number): void {
  for (const [key, times] of hits) {
    if (times[times.length - 1] <= windowStart) hits.delete(key);
  }
}
