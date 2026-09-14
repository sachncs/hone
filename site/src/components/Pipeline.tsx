import { motion } from "framer-motion";
import { Database, Cog, FileCheck2, Rocket } from "lucide-react";

const STEPS = [
  {
    n: "01",
    icon: Database,
    title: "Source",
    body:
      "Stream any HuggingFace dataset, point at a local JSONL, or mix in competitive-programming / SWE corpora.",
    code: [
      "from hone.prepare import prepare_stream, PrepareRequest",
      "request = PrepareRequest(output=Path(\"data/full/codex\"), seed=42)",
    ],
  },
  {
    n: "02",
    icon: Cog,
    title: "Normalize",
    body:
      "Auto-detect chat vs text format, normalize roles, drop malformed rows, apply tokenizer-based length cap.",
    code: [
      "prepare_stream(",
      "    request, repo=\"Modotte/CodeX-7M-Non-Thinking\",",
      "    mode=\"sft\", max_tokens=4096,",
      "    tokenizer_model=\"openbmb/MiniCPM5-1B\",",
      ")",
    ],
  },
  {
    n: "03",
    icon: FileCheck2,
    title: "Validate",
    body:
      "Deterministic seedable split. Reservoir sampling where needed. Named exceptions on every failure mode.",
    code: [
      "✓ accepted 92,041",
      "✓ dropped 2,034 (too long)",
      "✓ skipped 1,053 (malformed)",
    ],
  },
  {
    n: "04",
    icon: Rocket,
    title: "Train",
    body:
      "Write train.jsonl + valid.jsonl. Soup picks them up directly. One YAML, one trained model, one fused adapter.",
    code: [
      "$ uv run soup train \\",
      "    --config configs/soup-sft-codex-full.yaml",
    ],
  },
];

export function Pipeline() {
  return (
    <section id="pipeline" className="relative py-28 sm:py-36">
      <div aria-hidden className="absolute inset-x-0 top-0 -z-10 h-px bg-white/[0.06]" />
      <div aria-hidden className="absolute inset-0 -z-10 bg-mesh-section opacity-50" />

      <div className="container-page">
        <div className="mx-auto max-w-3xl text-center">
          <motion.span
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.5 }}
            className="text-eyebrow"
          >
            The pipeline
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="mt-4 text-display-lg gradient-text text-balance"
          >
            From raw corpus
            <br />
            to trained adapter.
          </motion.h2>
        </div>

        <div className="relative mx-auto mt-20 max-w-5xl">
          {/* Vertical line for desktop */}
          <div
            aria-hidden
            className="absolute left-[28px] top-2 hidden h-[calc(100%-2rem)] w-px bg-gradient-to-b from-white/[0.08] via-white/[0.04] to-transparent lg:left-1/2 lg:block lg:-translate-x-1/2"
          />

          <div className="space-y-10 lg:space-y-16">
            {STEPS.map((step, i) => {
              const side = i % 2 === 0 ? "left" : "right";
              return (
                <motion.div
                  key={step.n}
                  initial={{ opacity: 0, y: 24 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.3 }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                  className={`relative grid grid-cols-1 items-center gap-8 lg:grid-cols-2 ${
                    side === "right" ? "lg:[&>*:first-child]:order-2" : ""
                  }`}
                >
                  {/* node */}
                  <div className="absolute left-[18px] top-1 hidden h-5 w-5 lg:left-1/2 lg:block lg:-translate-x-1/2">
                    <div className="h-full w-full rounded-full border border-white/[0.1] bg-ink-950" />
                    <div className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent-400 shadow-[0_0_16px_rgba(94,133,255,0.6)]" />
                  </div>

                  <div className={`pl-16 lg:pl-0 ${side === "right" ? "lg:pl-12" : "lg:pr-12"}`}>
                    <div className="flex items-center gap-3">
                      <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-white/[0.06] bg-white/[0.03] text-ink-100 lg:hidden">
                        <step.icon className="h-4 w-4" strokeWidth={1.6} />
                      </span>
                      <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent-300">
                        Step {step.n}
                      </span>
                    </div>
                    <h3 className="mt-4 text-[26px] font-semibold tracking-tighter text-ink-50 sm:text-[30px]">
                      {step.title}
                    </h3>
                    <p className="mt-3 max-w-md text-[15px] leading-relaxed text-ink-300">
                      {step.body}
                    </p>
                  </div>

                  <div className={`pl-16 lg:pl-0 ${side === "right" ? "lg:pr-12" : "lg:pl-12"}`}>
                    <div className="surface overflow-hidden">
                      <div className="flex items-center justify-between border-b border-white/[0.05] px-4 py-2">
                        <div className="flex items-center gap-2 font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-400">
                          <span className="h-1.5 w-1.5 rounded-full bg-accent-400/70" />
                          {side === "left" ? "input" : "output"}
                        </div>
                        <span className="font-mono text-[10px] text-ink-500">.py</span>
                      </div>
                      <pre className="overflow-x-auto px-5 py-4 font-mono text-[12px] leading-[1.75] text-ink-100">
                        <code>
                          {step.code.map((l, j) => (
                            <div key={j} className="flex">
                              <span className="mr-4 inline-block w-5 select-none text-right text-ink-600">
                                {j + 1}
                              </span>
                              <span>{l}</span>
                            </div>
                          ))}
                        </code>
                      </pre>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
