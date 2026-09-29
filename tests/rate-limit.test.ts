import { describe, expect, it } from "vitest";

import { checkRateLimit } from "@/lib/rate-limit";

describe("checkRateLimit", () => {
  it("allows 50 requests per minute and blocks the 51st", () => {
    const now = 1_000_000;
    for (let i = 1; i <= 50; i++) {
      const result = checkRateLimit("ip-a", now + i);
      expect(result.allowed).toBe(true);
      expect(result.remaining).toBe(50 - i);
    }
    const blocked = checkRateLimit("ip-a", now + 51);
    expect(blocked.allowed).toBe(false);
    expect(blocked.remaining).toBe(0);
  });

  it("gives a Retry-After between 1 and 60 seconds", () => {
    const now = 2_000_000;
    for (let i = 0; i < 50; i++) checkRateLimit("ip-b", now);
    const blocked = checkRateLimit("ip-b", now + 15_000);
    expect(blocked.retryAfterSeconds).toBe(45);
  });

  it("slides the window: old requests expire after 60s", () => {
    const now = 3_000_000;
    for (let i = 0; i < 50; i++) checkRateLimit("ip-c", now);
    expect(checkRateLimit("ip-c", now + 59_999).allowed).toBe(false);
    expect(checkRateLimit("ip-c", now + 60_001).allowed).toBe(true);
  });

  it("tracks each IP separately", () => {
    const now = 4_000_000;
    for (let i = 0; i < 50; i++) checkRateLimit("ip-d", now);
    expect(checkRateLimit("ip-d", now).allowed).toBe(false);
    expect(checkRateLimit("ip-e", now).allowed).toBe(true);
  });
});
