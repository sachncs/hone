import { motion } from "framer-motion";
import { Layers, ShieldCheck, Cpu, Database, GitBranch, Sparkles } from "lucide-react";

const PILLARS = [
  {
    icon: Database,
    title: "One library. Every source.",
    body:
      "Local JSONL, HuggingFace streams, SWE-bench, CodeX, Nemotron-CP/SWE, Ling-Coder, LiveCodeBench eval prompts — all behind one consistent Python API.",
  },
  {
    icon: Layers,
    title: "Soup-ready, not \"almost-ready.\"",
    body:
      "Every row lands in `messages` chat format or `text` format. Token-length filtering is mandatory so the trainer never sees an empty loss target after truncation.",
  },
  {
    icon: ShieldCheck,
    title: "Deterministic by default.",
    body:
      "Seedable splits. Reservoir sampling that's actually uniform (Vitter algorithm R). Validation ratio is rejected if it's not in (0, 1) — never silently miscomputed.",
  },
  {
    icon: Cpu,
    title: "Honest about scale.",
    body:
      "Targets 1B-class SFT on an 18 GB M3 Pro, not 30B SOTA. We document the realistic ceiling and what would be needed to break it (see docs/SOTA-EXPECTATIONS.md).",
  },
  {
    icon: GitBranch,
    title: "Resumable across corpora.",
    body:
      "Compose CodeX → Ling-Coder → Nemotron → rStar-Coder in a single training run. Soup picks up each prepared JSONL and resumes the adapter.",
  },
  {
    icon: Sparkles,
    title: "Zero noise in the prep layer.",
    body:
      "No MLX, no Unsloth, no CUDA. The `hone` package is `datasets` + Python stdlib. Your training backend is your choice.",
  },
];

export function Overview() {
  return (
    <section id="product" className="relative py-28 sm:py-36">
      <div aria-hidden className="absolute inset-0 -z-10 bg-mesh-section opacity-60" />
      <div className="container-page">
        <div className="grid grid-cols-1 items-end gap-10 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <motion.span
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.5 }}
              className="text-eyebrow"
            >
              The product
            </motion.span>
            <motion.h2
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              className="mt-4 text-display-lg gradient-text text-balance"
            >
              Everything between
              <br />
              <span className="gradient-accent">a raw corpus</span> and a
              <br />
              trained adapter.
            </motion.h2>
          </div>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
            className="max-w-md text-[16px] leading-relaxed text-ink-300"
          >
            hone is the missing layer between HuggingFace and Soup. It does one thing — produce a
            deterministic, length-filtered, chat-formatted JSONL that your trainer will not choke on
            — and does it without the runtime bloat of a full pipeline.
          </motion.p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-px overflow-hidden rounded-3xl border border-white/[0.06] bg-white/[0.04] md:grid-cols-2 lg:grid-cols-3">
          {PILLARS.map((p, i) => (
            <motion.div
              key={p.title}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1], delay: i * 0.05 }}
              className="group relative bg-ink-975/80 p-7 transition-colors duration-300 hover:bg-ink-900/80"
            >
              <div className="flex items-center gap-3">
                <span className="grid h-9 w-9 place-items-center rounded-xl border border-white/[0.06] bg-white/[0.03] text-ink-100 transition-colors group-hover:border-accent-400/40 group-hover:text-accent-200">
                  <p.icon className="h-4 w-4" strokeWidth={1.6} />
                </span>
                <span className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-500">
                  0{i + 1}
                </span>
              </div>
              <h3 className="mt-5 text-[17px] font-semibold tracking-tight text-ink-50">
                {p.title}
              </h3>
              <p className="mt-2 text-[13.5px] leading-relaxed text-ink-300">{p.body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
