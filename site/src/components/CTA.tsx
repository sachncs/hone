import { motion } from "framer-motion";
import { ArrowRight, Github, Terminal, Copy, Check } from "lucide-react";
import { useState } from "react";

const INSTALL = `# clone & bootstrap
git clone https://github.com/sachncs/hone.git
cd hone
./setup.sh              # venv + hone[dev,soup] + pytest + smoke

# or, step by step:
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
uv pip install "soup-cli[mlx]==0.73.2" \\
                "transformers>=4.57,<5" \\
                "huggingface-hub<1.0,>=0.34"`;

export function CTA() {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(INSTALL);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* noop */
    }
  };

  return (
    <section id="install" className="relative py-28 sm:py-36">
      <div className="container-page">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className="surface-deep relative overflow-hidden"
        >
          {/* Decorative */}
          <div
            aria-hidden
            className="pointer-events-none absolute -top-40 left-1/2 h-[420px] w-[820px] -translate-x-1/2 rounded-full bg-accent-500/15 blur-[120px]"
          />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 grid-pattern opacity-30"
          />

          <div className="relative grid grid-cols-1 gap-10 p-8 sm:p-12 lg:grid-cols-[1.1fr_1fr] lg:p-16">
            <div>
              <span className="text-eyebrow">Get started</span>
              <h2 className="mt-4 text-display-lg gradient-text text-balance">
                Train a better
                <br />
                <span className="gradient-accent">small model</span>
                <br />
                this week.
              </h2>
              <p className="mt-5 max-w-md text-[15.5px] leading-relaxed text-ink-300">
                Bootstrap takes ~90 seconds. Soup smoke run takes ~30 seconds and validates the
                end-to-end pipeline against a 40-row fixture. Full CodeX SFT for ~70 minutes on an
                M3 Pro 18 GB.
              </p>

              <div className="mt-8 flex flex-wrap items-center gap-3">
                <a href="https://github.com/sachncs/hone" target="_blank" rel="noreferrer" className="btn-primary group px-6 py-3">
                  <Github className="h-4 w-4" strokeWidth={1.8} />
                  <span>Star on GitHub</span>
                  <ArrowRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5" />
                </a>
                <a href="https://github.com/sachncs/hone#quick-start" target="_blank" rel="noreferrer" className="btn-secondary px-6 py-3">
                  <Terminal className="h-4 w-4" strokeWidth={1.8} />
                  <span>Read the docs</span>
                </a>
              </div>

              <div className="mt-8 flex items-center gap-6 text-[12px] text-ink-400">
                <div className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  <span>MIT licensed</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-accent-400" />
                  <span>Python 3.12 / 3.13</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-accent-400" />
                  <span>Apple Silicon native</span>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="surface overflow-hidden">
                <div className="flex items-center justify-between border-b border-white/[0.05] px-4 py-2.5">
                  <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
                    <span className="h-1.5 w-1.5 rounded-full bg-accent-400/70" />
                    quickstart · zsh
                  </div>
                  <button
                    onClick={onCopy}
                    className="inline-flex items-center gap-1.5 rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 py-1 font-mono text-[11px] text-ink-300 transition-colors hover:bg-white/[0.05] hover:text-ink-100"
                    aria-label="Copy install command"
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
                <pre className="overflow-x-auto px-5 py-5 font-mono text-[12.5px] leading-[1.8] text-ink-100">
                  <code>
                    {INSTALL.split("\n").map((l, i) => (
                      <div key={i} className="flex">
                        <span className="mr-4 inline-block w-5 select-none text-right text-ink-600">
                          {i + 1}
                        </span>
                        <span
                          className={
                            l.trimStart().startsWith("#")
                              ? "text-ink-500 italic"
                              : l.trimStart().startsWith("uv")
                                ? "text-accent-300"
                                : "text-ink-100"
                          }
                        >
                          {l || "\u00A0"}
                        </span>
                      </div>
                    ))}
                  </code>
                </pre>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
