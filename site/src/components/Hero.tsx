import { motion } from "framer-motion";
import { ArrowRight, Sparkles, Terminal, Zap } from "lucide-react";
import { HeroConsole } from "./HeroConsole";

export function Hero() {
  return (
    <section id="top" className="relative isolate overflow-hidden pt-32 pb-24 sm:pt-40 sm:pb-32">
      {/* Layered backgrounds */}
      <div aria-hidden className="absolute inset-0 -z-20 bg-mesh-hero" />
      <div aria-hidden className="absolute inset-x-0 top-0 -z-10 h-[1200px] grid-pattern opacity-60" />
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 -z-10 h-[800px] bg-gradient-to-b from-transparent via-transparent to-ink-975"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-40 -z-10 h-[420px] w-[820px] -translate-x-1/2 rounded-full bg-accent-500/15 blur-[120px]"
      />

      <div className="container-page">
        {/* Eyebrow */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto flex w-fit items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 backdrop-blur"
        >
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
          </span>
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-200">
            v0.3.0 · Soup-first release
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.05 }}
          className="mx-auto mt-8 max-w-4xl text-center text-display-2xl gradient-text text-balance"
        >
          The precision layer
          <br />
          for Soup-driven
          <br />
          <span className="gradient-accent">fine-tuning.</span>
        </motion.h1>

        {/* Subhead */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.18 }}
          className="mx-auto mt-7 max-w-2xl text-center text-[17px] leading-relaxed text-ink-300 sm:text-[18.5px] text-balance"
        >
          hone prepares the JSONL your trainer will actually consume — chat-formatted, length-filtered,
          reservoir-sampled, split, and Soup-ready. One Python library. One YAML. One trained model on
          your MacBook.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.28 }}
          className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row"
        >
          <a href="#install" className="btn-primary group px-6 py-3 text-[14px]">
            <Terminal className="h-4 w-4" strokeWidth={2} />
            <span>Install in 60 seconds</span>
            <ArrowRight className="h-4 w-4 transition-transform duration-300 ease-smooth group-hover:translate-x-0.5" />
          </a>
          <a
            href="https://github.com/sachncs/hone"
            target="_blank"
            rel="noreferrer"
            className="btn-secondary px-6 py-3 text-[14px]"
          >
            <span>View on GitHub</span>
          </a>
        </motion.div>

        {/* Tech chips */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.45 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-2"
        >
          {[
            { icon: Sparkles, label: "Apple Silicon native" },
            { icon: Zap, label: "7.4 GB peak · M3 Pro 18 GB" },
            { icon: Terminal, label: "Zero MLX deps in the prep layer" },
          ].map(({ icon: Icon, label }) => (
            <span key={label} className="chip">
              <Icon className="h-3 w-3 text-accent-300" strokeWidth={2} />
              <span>{label}</span>
            </span>
          ))}
        </motion.div>

        {/* Hero console */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.4 }}
          className="mx-auto mt-20 max-w-5xl"
        >
          <HeroConsole />
        </motion.div>

        {/* Soft fade bottom */}
        <div aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 h-32 bg-gradient-to-b from-transparent to-ink-975" />
      </div>
    </section>
  );
}
