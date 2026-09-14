import { motion } from "framer-motion";
import { TrendingUp, Clock, Cpu, Database, ChartLine } from "lucide-react";

const METRICS = [
  { value: "7.4", unit: "GB", label: "Peak memory", sub: "M3 Pro 18 GB · seq 2048 · batch 1", icon: Cpu },
  { value: "880", unit: "tok/s", label: "Throughput", sub: "MLX · bf16 · grad checkpointing", icon: TrendingUp },
  { value: "~70", unit: "min", label: "Full SFT", sub: "1 epoch over 95k CodeX rows", icon: Clock },
  { value: "12", unit: "", label: "End-to-end tests", sub: "Splits, filters, exceptions, mappers", icon: Database },
];

const TABLE = [
  { name: "Base MiniCPM5-1B-MLX", he: 39.6, mbpp: 28.0, iters: "—", note: "Out-of-the-box" },
  { name: "5K SFT · CodeX + Ling", he: 38.4, mbpp: 30.0, iters: "4,750", note: "Smoke-scale" },
  { name: "16K SFT (partial)", he: 29.3, mbpp: 33.0, iters: "6,200 / 15.2K", note: "Killed at 41%" },
];

const MAX_HE = 70;

export function Benchmarks() {
  return (
    <section id="benchmarks" className="relative py-28 sm:py-36">
      <div aria-hidden className="absolute inset-x-0 top-0 -z-10 h-px bg-white/[0.06]" />
      <div className="container-page">
        <div className="grid grid-cols-1 items-end gap-10 lg:grid-cols-[1fr_1.1fr]">
          <div>
            <motion.span
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.5 }}
              className="text-eyebrow"
            >
              Measured, not promised
            </motion.span>
            <motion.h2
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              className="mt-4 text-display-lg gradient-text text-balance"
            >
              Honest numbers
              <br />
              from an honest ceiling.
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
              className="mt-5 max-w-md text-[16px] leading-relaxed text-ink-300"
            >
              Every figure below comes from <code className="font-mono text-[13px] text-ink-100">python -m bench</code> — greedy decoding, sandboxed test
              subprocesses with a 2–5s timeout. Reproduce with the exact configs in
              <code className="font-mono text-[13px] text-ink-100"> configs/</code>.
            </motion.p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-2">
            {METRICS.map((m, i) => (
              <motion.div
                key={m.label}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1], delay: i * 0.06 }}
                className="surface relative overflow-hidden p-5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-400">
                    {m.label}
                  </span>
                  <m.icon className="h-3.5 w-3.5 text-ink-400" strokeWidth={1.7} />
                </div>
                <div className="mt-4 flex items-baseline gap-1">
                  <span className="text-[34px] font-semibold tracking-tighter text-ink-50">
                    {m.value}
                  </span>
                  <span className="font-mono text-[12px] text-ink-400">{m.unit}</span>
                </div>
                <p className="mt-1.5 text-[12px] leading-relaxed text-ink-400">{m.sub}</p>
                <div
                  aria-hidden
                  className="pointer-events-none absolute -bottom-12 -right-10 h-32 w-32 rounded-full bg-accent-500/10 blur-2xl"
                />
              </motion.div>
            ))}
          </div>
        </div>

        {/* Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="surface-deep mt-16 overflow-hidden"
        >
          <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
            <div className="flex items-center gap-2">
              <ChartLine className="h-3.5 w-3.5 text-accent-300" strokeWidth={1.8} />
              <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
                ablation · MiniCPM5-1B-MLX
              </span>
            </div>
            <span className="font-mono text-[10.5px] text-ink-500">results/</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr]">
            <div className="border-b border-white/[0.06] lg:border-b-0 lg:border-r">
              {/* Header */}
              <div className="grid grid-cols-12 border-b border-white/[0.04] px-6 py-3 font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-400">
                <div className="col-span-5">Run</div>
                <div className="col-span-2 text-right">HE</div>
                <div className="col-span-2 text-right">MBPP</div>
                <div className="col-span-3 text-right">iters</div>
              </div>
              {TABLE.map((row, i) => (
                <div
                  key={row.name}
                  className={`grid grid-cols-12 items-center px-6 py-4 ${
                    i < TABLE.length - 1 ? "border-b border-white/[0.04]" : ""
                  }`}
                >
                  <div className="col-span-5">
                    <div className="text-[13.5px] font-medium text-ink-50">{row.name}</div>
                    <div className="mt-0.5 font-mono text-[10.5px] text-ink-500">{row.note}</div>
                  </div>
                  <div className="col-span-2 text-right">
                    <Bar value={row.he} max={MAX_HE} tone="accent" />
                    <div className="mt-1 font-mono text-[12px] text-ink-100">{row.he}%</div>
                  </div>
                  <div className="col-span-2 text-right">
                    <Bar value={row.mbpp} max={70} tone="muted" />
                    <div className="mt-1 font-mono text-[12px] text-ink-100">{row.mbpp}%</div>
                  </div>
                  <div className="col-span-3 text-right font-mono text-[12px] text-ink-300">
                    {row.iters}
                  </div>
                </div>
              ))}
            </div>

            <div className="p-6 sm:p-8">
              <div className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-ink-400">
                Reality check
              </div>
              <h3 className="mt-3 text-[22px] font-semibold tracking-tight text-ink-50">
                1B SFT ≠ frontier SOTA.
              </h3>
              <p className="mt-3 text-[13.5px] leading-relaxed text-ink-300">
                A single 18 GB MacBook cannot close the gap to DeepSeek V4-Flash (284B MoE, 13B
                active) by throwing more LoRA data at it. The architectural ceiling is
                architecture — not data.
              </p>
              <p className="mt-3 text-[13.5px] leading-relaxed text-ink-300">
                The realistic ceiling for an honest 1B LoRA on this hardware is{" "}
                <span className="text-ink-50">45–55% HumanEval</span> and{" "}
                <span className="text-ink-50">55–65% MBPP</span>, matching Qwen2.5-Coder-1.5B /
                DeepSeek-Coder-1.3B. To go beyond that, the path is teacher distillation on a 24+
                GB GPU host — Soup supports this directly via{" "}
                <code className="font-mono text-[12.5px] text-accent-200">distill-prompt</code>.
              </p>
              <div className="mt-6 flex flex-wrap gap-2">
                <a href="#docs" className="btn-secondary text-[12.5px]">
                  Read SOTA-EXPECTATIONS.md
                </a>
                <a href="#docs" className="btn-ghost text-[12.5px]">
                  Read ABLATION-REPORT.md →
                </a>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function Bar({ value, max, tone }: { value: number; max: number; tone: "accent" | "muted" }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="ml-auto h-1 w-20 overflow-hidden rounded-full bg-white/[0.06]">
      <motion.div
        initial={{ width: 0 }}
        whileInView={{ width: `${pct}%` }}
        viewport={{ once: true, amount: 0.5 }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        className={
          tone === "accent"
            ? "h-full rounded-full bg-gradient-to-r from-accent-400 to-accent-200"
            : "h-full rounded-full bg-gradient-to-r from-steel-400 to-steel-300"
        }
      />
    </div>
  );
}
