import { useEffect, useState } from "react";
import { cn } from "../lib/utils";

const TYPED_LINES: { type: "comment" | "import" | "code" | "blank"; text: string }[] = [
  { type: "comment", text: "# prepare a Soup-ready JSONL from CodeX" },
  { type: "import", text: "from hone.prepare import prepare_stream, PrepareRequest" },
  { type: "blank", text: "" },
  { type: "comment", text: "# chat format · length cap · seedable split" },
  { type: "code", text: "request = PrepareRequest(output=Path(\"data/full/codex\"), seed=42)" },
  { type: "blank", text: "" },
  { type: "code", text: "prepare_stream(" },
  { type: "code", text: "    request=request," },
  { type: "code", text: "    repo=\"Modotte/CodeX-7M-Non-Thinking\"," },
  { type: "code", text: "    configs=\"default\"," },
  { type: "code", text: "    mode=\"sft\"," },
  { type: "code", text: "    max_tokens=4096," },
  { type: "code", text: "    tokenizer_model=\"openbmb/MiniCPM5-1B\"," },
  { type: "code", text: ")" },
];

const LOG_LINES = [
  { tone: "muted", text: "→ streaming Modotte/CodeX-7M-Non-Thinking · config=default" },
  { tone: "muted", text: "→ chat format detected · 95,128 rows" },
  { tone: "muted", text: "→ token filter · model=openbmb/MiniCPM5-1B · cap=4096" },
  { tone: "ok", text: "✓ accepted 92,041  dropped 2,034 (too long)  skipped 1,053 (malformed)" },
  { tone: "muted", text: "→ split · seed=42 · ratio=0.02 → 90,200 train / 1,841 valid" },
  { tone: "ok", text: "✓ wrote data/full/codex/train.jsonl  (164.2 MB)" },
  { tone: "ok", text: "✓ wrote data/full/codex/valid.jsonl  (  3.3 MB)" },
] as const;

export function HeroConsole() {
  const [progress, setProgress] = useState(0);
  const [showLogs, setShowLogs] = useState(false);
  const [logIndex, setLogIndex] = useState(0);

  useEffect(() => {
    const totalChars = TYPED_LINES.reduce(
      (acc, l) => acc + (l.type === "blank" ? 1 : l.text.length + 1),
      0,
    );
    let i = 0;
    const tick = () => {
      i += 1;
      setProgress(Math.min(i / totalChars, 1));
      if (i >= totalChars) {
        clearInterval(t);
        setTimeout(() => setShowLogs(true), 280);
      }
    };
    const t = setInterval(tick, 16);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!showLogs) return;
    setLogIndex(0);
    const t = setInterval(() => {
      setLogIndex((i) => {
        if (i >= LOG_LINES.length - 1) {
          clearInterval(t);
          return i;
        }
        return i + 1;
      });
    }, 420);
    return () => clearInterval(t);
  }, [showLogs]);

  // Compute cumulative character offsets so the typewriter reveals each
  // line one after another instead of all at once.
  let consumed = 0;
  const rendered = TYPED_LINES.map((line, idx) => {
    const lineLen = line.type === "blank" ? 1 : line.text.length + 1;
    const start = consumed;
    const end = consumed + lineLen;
    consumed = end;

    const visibleChars = Math.max(
      0,
      Math.min(
        line.text.length,
        Math.floor((progress * TYPED_LINES.reduce((acc, l) => acc + (l.type === "blank" ? 1 : l.text.length + 1), 0)) - start),
      ),
    );

    return {
      idx,
      type: line.type,
      visible: start <= progress * TYPED_LINES.reduce((acc, l) => acc + (l.type === "blank" ? 1 : l.text.length + 1), 0),
      text: line.text.slice(0, visibleChars),
    };
  });

  return (
    <div className="relative">
      {/* glow */}
      <div
        aria-hidden
        className="absolute -inset-8 -z-10 rounded-[32px] bg-gradient-to-br from-accent-500/20 via-accent-400/10 to-transparent blur-3xl"
      />

      <div className="surface-deep shimmer-border overflow-hidden">
        {/* Window chrome */}
        <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]/80" />
          </div>
          <div className="flex items-center gap-2 rounded-md border border-white/5 bg-white/[0.02] px-2.5 py-1 font-mono text-[11px] text-ink-400">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-400" />
            <span>~/hone · uv run python</span>
          </div>
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
            <span>prepare</span>
          </div>
        </div>

        {/* Code area */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1px_1fr]">
          <div className="relative">
            <div className="flex items-center justify-between border-b border-white/[0.04] px-5 py-2">
              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-accent-400/70" />
                <span className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-300">
                  prepare_stream.py
                </span>
              </div>
              <span className="font-mono text-[10px] text-ink-500">python 3.12</span>
            </div>
            <pre className="overflow-x-auto px-5 py-4 font-mono text-[12.5px] leading-[1.7] text-ink-100">
              <code>
                {rendered.map((l) => (
                  <div key={l.idx} className="flex">
                    <span className="mr-4 inline-block w-6 select-none text-right text-ink-600">
                      {l.idx + 1}
                    </span>
                    <span
                      className={cn(
                        l.type === "comment" && "text-ink-400 italic",
                        l.type === "import" && "text-accent-300",
                        l.type === "code" && "text-ink-100",
                        l.type === "blank" && "h-4",
                      )}
                    >
                      {l.text}
                      <span className="ml-[1px] inline-block h-[1em] w-[6px] -mb-[2px] animate-blink-soft bg-accent-300 align-middle" />
                    </span>
                  </div>
                ))}
              </code>
            </pre>
          </div>

          <div className="hidden bg-white/[0.04] lg:block" />

          {/* Logs / output */}
          <div className="relative bg-ink-975/40">
            <div className="flex items-center justify-between border-b border-white/[0.04] px-5 py-2">
              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400/80" />
                <span className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-300">
                  output
                </span>
              </div>
              <span className="font-mono text-[10px] text-ink-500">stdout</span>
            </div>
            <div className="px-5 py-4 font-mono text-[12px] leading-[1.8] text-ink-200">
              {showLogs ? (
                LOG_LINES.slice(0, logIndex + 1).map((l, i) => (
                  <div
                    key={i}
                    className={cn(
                      "animate-fade-up",
                      l.tone === "ok" ? "text-emerald-300/90" : "text-ink-300",
                    )}
                  >
                    {l.text}
                  </div>
                ))
              ) : (
                <div className="text-ink-500">awaiting output…</div>
              )}
            </div>

            {/* Result pill */}
            {logIndex >= LOG_LINES.length - 1 && (
              <div className="mx-5 mb-5 mt-1 flex animate-fade-up items-center justify-between rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-400">
                    summary
                  </div>
                  <div className="mt-0.5 text-[12px] text-ink-100">
                    92,041 rows · Soup-ready
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-400">
                    wall
                  </div>
                  <div className="mt-0.5 font-mono text-[12px] text-ink-100">3m 18s</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
