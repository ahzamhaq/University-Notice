"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

import styles from "./SearchNotices.module.css";

interface Source {
  title: string;
  url: string;
  noticeDate: string | null;
  similarity: number;
  /** Relevance rank (1 = best); the answer cites notices by this number. */
  ref: number;
  /** True for the notices the answer was written from. */
  cited: boolean;
}

type SortOrder = "newest" | "relevance";

/** The API sends sources in relevance order; "newest" re-sorts by date (undated last). */
function sortSources(sources: Source[], order: SortOrder): Source[] {
  if (order === "relevance") return sources;
  return [...sources].sort((a, b) => (b.noticeDate ?? "").localeCompare(a.noticeDate ?? ""));
}

type StreamEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; error: string };

type Phase = "idle" | "searching" | "answering" | "done";

const EXAMPLES = [
  "When is the next datesheet for LLB exams?",
  "Are there any sports selection trials this month?",
  "How do I apply for the NSP scholarship?",
];

function formatDate(iso: string | null): string | null {
  if (!iso) return null;
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

export default function SearchNotices() {
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[] | null>(null);
  const [answer, setAnswer] = useState("");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");
  const [showAll, setShowAll] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Cancel generation if the component unmounts (e.g. the tab is closed).
  useEffect(() => () => abortRef.current?.abort(), []);

  const busy = phase === "searching" || phase === "answering";

  // The answer cites the notices the model read ([1]..[5]); the rest appear with "show more".
  const citedSources = sources?.filter((s) => s.cited) ?? [];
  const extraCount = (sources?.length ?? 0) - citedSources.length;
  const visibleSources = sortSources(showAll ? (sources ?? []) : citedSources, sortOrder);

  async function ask(q: string) {
    const trimmed = q.trim();
    if (trimmed.length < 3) return;

    // A new question cancels the previous one, freeing the model on the server.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("searching");
    setError(null);
    setSources(null);
    setAnswer("");
    setShowAll(false);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
        body: JSON.stringify({ question: trimmed }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line) as StreamEvent;
          if (event.type === "sources") {
            setSources(event.sources);
            setPhase("answering");
          } else if (event.type === "token") {
            setAnswer((prev) => prev + event.text);
          } else if (event.type === "error") {
            throw new Error(event.error);
          }
        }
      }
      setPhase("done");
    } catch (err) {
      if (controller.signal.aborted) return; // superseded by a newer question
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setPhase("done");
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void ask(question);
  }

  return (
    <section className={styles.wrapper}>
      <form className={styles.form} onSubmit={onSubmit} role="search">
        <label htmlFor="notice-question" className={styles.srOnly}>
          Ask about university notices
        </label>
        <input
          id="notice-question"
          className={styles.input}
          type="search"
          placeholder="Ask about exams, admissions, scholarships, sports…"
          value={question}
          maxLength={500}
          onChange={(e) => setQuestion(e.target.value)}
          autoFocus
        />
        <button className={styles.button} type="submit" disabled={question.trim().length < 3}>
          {busy ? "Ask again" : "Ask"}
        </button>
      </form>

      {phase === "idle" && (
        <div className={styles.examples}>
          <span>Try:</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className={styles.chip}
              onClick={() => {
                setQuestion(example);
                void ask(example);
              }}
            >
              {example}
            </button>
          ))}
        </div>
      )}

      {phase === "searching" && (
        <p className={styles.status} aria-live="polite">
          Searching the notices…
        </p>
      )}

      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}

      {sources && (
        <article className={styles.result}>
          <h2 className={styles.heading}>Answer</h2>
          <p className={styles.answer} aria-live="polite" aria-busy={phase === "answering"}>
            {answer}
            {phase === "answering" && (
              <span className={styles.writing}>
                {answer ? "▍" : "Reading the notices below and writing an answer — this can take a minute…"}
              </span>
            )}
          </p>

          {sources.length > 0 && (
            <>
              <div className={styles.sourcesHeader}>
                <h2 className={styles.heading}>Matching notices</h2>
                <div className={styles.sortToggle} role="group" aria-label="Sort notices">
                  <button type="button" aria-pressed={sortOrder === "newest"} onClick={() => setSortOrder("newest")}>
                    Newest first
                  </button>
                  <button
                    type="button"
                    aria-pressed={sortOrder === "relevance"}
                    onClick={() => setSortOrder("relevance")}
                  >
                    Best match
                  </button>
                </div>
              </div>
              <ul className={styles.sources}>
                {visibleSources.map((source, i) => (
                  <li key={source.url}>
                    <span className={styles.position}>{i + 1}.</span>
                    <div>
                      <a href={source.url} target="_blank" rel="noopener noreferrer">
                        {source.title}
                      </a>
                      <span className={styles.meta}>
                        {formatDate(source.noticeDate) ?? "Undated"} · {Math.round(source.similarity * 100)}% match
                        {source.cited && (
                          <span className={styles.cited} title={`The answer refers to this notice as [${source.ref}]`}>
                            cited [{source.ref}]
                          </span>
                        )}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
              {extraCount > 0 && (
                <button type="button" className={styles.more} onClick={() => setShowAll((v) => !v)}>
                  {showAll
                    ? "Show fewer"
                    : `Show ${extraCount} more matching notice${extraCount === 1 ? "" : "s"}`}
                </button>
              )}
            </>
          )}
        </article>
      )}
    </section>
  );
}
