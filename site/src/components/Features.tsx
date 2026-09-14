import { motion } from "framer-motion";
import {
  Wand2,
  Filter,
  Shuffle,
  Split,
  CheckCircle2,
  Binary,
  TestTube2,
  Boxes,
} from "lucide-react";

const FEATURES = [
  {
    icon: Wand2,
    eyebrow: "Format",
    title: "Auto-detect chat vs text",
    body:
      "Reads each row, normalizes into either `messages` chat format or `text` format. Handles OpenAI/HF templates, SWE-bench patches, codeforces prompts.",
  },
  {
    icon: Filter,
    eyebrow: "Length",
    title: "Tokenizer-based filtering",
    body:
      "Drop rows that exceed `max_tokens` so the trainer never sees an empty loss target after truncation. Uses any HF tokenizer — defaults to MiniCPM5-1B.",
  },
  {
    icon: Shuffle,
    eyebrow: "Sampling",
    title: "Vitter reservoir sampling",
    body:
      "Uniform-random sampling of arbitrarily large HF corpora with O(1) memory per row. Algorithm R, deterministic given seed.",
  },
  {
    icon: Split,
    eyebrow: "Splits",
    title: "Seedable train / valid",
    body:
      "Reproducible splits. Ratio validated to be exclusive (0, 1). Empty rows, malformed JSON, and bad chat roles are rejected with named exceptions.",
  },
  {
    icon: Binary,
    eyebrow: "Throughput",
    title: "Streaming, not loading",
    body:
      "Streams HuggingFace datasets row-by-row. Materializes only the rows that pass filters. 95k-row CodeX corpus fits in ~3 minutes on an M3 Pro.",
  },
  {
    icon: Boxes,
    eyebrow: "Sources",
    title: "Seven prep recipes",
    body:
      "Local JSONL, generic HF stream, CodeX, Ling-Coder, Nemotron-CP + SWE, SWE-bench, LiveCodeBench prompts. One Python API for each.",
  },
  {
    icon: CheckCircle2,
    eyebrow: "Errors",
    title: "Named exception hierarchy",
    body:
      "PrepareError, DataError, ValidationError — catch every prepare-layer failure with one `except`. No silent failures, no stack traces in stdout.",
  },
  {
    icon: TestTube2,
    eyebrow: "Tests",
    title: "12 end-to-end tests",
    body:
      "Splits, malformed JSON, ratio bounds, role validation, Nemotron row normalizer, exception hierarchy. `pytest` clean on Linux CI.",
  },
];

export function Features() {
  return (
    <section id="features" className="relative py-28 sm:py-36">
      <div className="container-page">
        <div className="mx-auto max-w-3xl text-center">
          <motion.span
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.5 }}
            className="text-eyebrow"
          >
            Capabilities
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="mt-4 text-display-lg gradient-text text-balance"
          >
            Built for the boring parts of ML.
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
            className="mt-5 text-[16.5px] leading-relaxed text-ink-300"
          >
            The data layer is where models are quietly won or lost. hone makes that layer boring on
            purpose.
          </motion.p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1], delay: i * 0.04 }}
              className="group relative flex flex-col rounded-2xl border border-white/[0.06] bg-gradient-to-b from-white/[0.025] to-white/[0.005] p-6 transition-all duration-300 hover:border-white/[0.1] hover:from-white/[0.04] hover:to-white/[0.01]"
            >
              <div className="mb-5 flex items-center justify-between">
                <span className="grid h-9 w-9 place-items-center rounded-xl border border-white/[0.06] bg-white/[0.03] text-ink-100 transition-colors group-hover:text-accent-200">
                  <f.icon className="h-4 w-4" strokeWidth={1.6} />
                </span>
                <span className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-500">
                  {f.eyebrow}
                </span>
              </div>
              <h3 className="text-[15px] font-semibold tracking-tight text-ink-50">{f.title}</h3>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-300">{f.body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
