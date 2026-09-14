import { motion } from "framer-motion";
import { Copy, Check, Terminal } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const TABS = [
  {
    id: "stream",
    label: "prepare_stream",
    caption: "HuggingFace → chat JSONL",
    code: `from pathlib import Path
import logging
from hone.prepare import prepare_stream, PrepareRequest

logging.basicConfig(level=logging.INFO)
request = PrepareRequest(output=Path("data/full/codex"), seed=42)

prepare_stream(
    request=request,
    repo="Modotte/CodeX-7M-Non-Thinking",
    configs="default",
    mode="sft",
    max_tokens=4096,
    tokenizer_model="openbmb/MiniCPM5-1B",
)`,
  },
  {
    id: "nemotron",
    label: "prepare_nemotron",
    caption: "Bypass HF CastError",
    code: `from hone.prepare import prepare_nemotron, PrepareRequest
from hone.prepare.nemotron import NemotronConfig

request = PrepareRequest(output=Path("data/soup/nemotron-combined"), seed=42)

prepare_nemotron(
    request,
    configs=[
        NemotronConfig(split="Competitive-Programming-v2", cap=4096),
        NemotronConfig(split="SWE-v2", cap=4096),
    ],
    tokenizer_model="openbmb/MiniCPM5-1B",
)`,
  },
  {
    id: "reservoir",
    label: "prepare_reservoir_sample",
    caption: "Vitter-R, deterministic",
    code: `from hone.prepare import prepare_reservoir_sample, PrepareRequest

request = PrepareRequest(output=Path("data/full/codeforces"), seed=42)

prepare_reservoir_sample(
    request=request,
    repo="open-r1/codeforces",
    max_samples=50_000,
    max_tokens=2048,
    tokenizer_model="openbmb/MiniCPM5-1B",
)`,
  },
  {
    id: "ling",
    label: "prepare_ling_coder",
    caption: "Ling-Coder SFT stream",
    code: `from hone.prepare import prepare_ling_coder, PrepareRequest

request = PrepareRequest(output=Path("data/full/ling-coder"), seed=42)

prepare_ling_coder(
    request=request,
    max_tokens=4096,
    tokenizer_model="openbmb/MiniCPM5-1B",
)`,
  },
];

export function CodePreview() {
  const [active, setActive] = useState(TABS[0].id);
  const [copied, setCopied] = useState(false);
  const tab = TABS.find((t) => t.id === active)!;

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(tab.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* noop */
    }
  };

  return (
    <section id="docs" className="relative py-28 sm:py-36">
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
            The surface
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="mt-4 text-display-lg gradient-text text-balance"
          >
            A small API on purpose.
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
            className="mt-5 text-[16.5px] leading-relaxed text-ink-300"
          >
            Seven prep functions. One <code className="font-mono text-[14px] text-ink-100">PrepareRequest</code>.
            Every row that lands on disk has been normalized, length-filtered, and split with a
            seed you can replay.
          </motion.p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="surface-deep relative mx-auto mt-14 max-w-5xl overflow-hidden"
        >
          <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-3">
            <div className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]/80" />
              <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]/80" />
              <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]/80" />
            </div>
            <div className="flex items-center gap-2 rounded-md border border-white/5 bg-white/[0.02] px-2.5 py-1 font-mono text-[11px] text-ink-400">
              <Terminal className="h-3 w-3" strokeWidth={1.8} />
              <span>hone.prepare</span>
            </div>
            <button
              onClick={onCopy}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 py-1 font-mono text-[11px] text-ink-300 transition-colors hover:bg-white/[0.05] hover:text-ink-100",
              )}
              aria-label="Copy code"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 text-emerald-300" strokeWidth={2} /> Copied
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3" strokeWidth={1.7} /> Copy
                </>
              )}
            </button>
          </div>

          <div className="border-b border-white/[0.06] px-4 py-2">
            <div className="flex flex-wrap items-center gap-1">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActive(t.id)}
                  className={cn(
                    "group inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 font-mono text-[12px] transition-all duration-200",
                    active === t.id
                      ? "bg-white/[0.06] text-ink-50"
                      : "text-ink-400 hover:bg-white/[0.03] hover:text-ink-100",
                  )}
                >
                  <span
                    className={cn(
                      "h-1.5 w-1.5 rounded-full transition-colors",
                      active === t.id ? "bg-accent-400" : "bg-white/20",
                    )}
                  />
                  {t.label}
                </button>
              ))}
            </div>
            <div className="mt-2 px-1 font-mono text-[11px] text-ink-500">{tab.caption}</div>
          </div>

          <pre className="overflow-x-auto px-6 py-6 font-mono text-[13px] leading-[1.85] text-ink-100">
            <code>
              {tab.code.split("\n").map((l, i) => (
                <motion.div
                  key={`${active}-${i}`}
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{
                    duration: 0.3,
                    ease: [0.22, 1, 0.36, 1],
                    delay: i * 0.025,
                  }}
                  className="flex"
                >
                  <span className="mr-5 inline-block w-6 select-none text-right text-ink-600">
                    {i + 1}
                  </span>
                  <span
                    className={cn(
                      l.trimStart().startsWith("from") || l.trimStart().startsWith("import")
                        ? "text-accent-300"
                        : l.trimStart().startsWith("#")
                          ? "text-ink-500 italic"
                          : "text-ink-100",
                    )}
                  >
                    {l || "\u00A0"}
                  </span>
                </motion.div>
              ))}
            </code>
          </pre>
        </motion.div>

        {/* Public surface strip */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto mt-10 grid max-w-5xl grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5"
        >
          {[
            "prepare_local_file",
            "prepare_stream",
            "prepare_reservoir_sample",
            "prepare_swe",
            "prepare_ling_coder",
            "prepare_nemotron",
            "prepare_eval_prompts",
            "PrepareRequest",
            "PrepareResult",
            "PrepareError",
          ].map((fn) => (
            <div
              key={fn}
              className="flex items-center gap-2 rounded-xl border border-white/[0.05] bg-white/[0.02] px-3 py-2"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-accent-400/70" />
              <span className="font-mono text-[11.5px] text-ink-200">{fn}</span>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
