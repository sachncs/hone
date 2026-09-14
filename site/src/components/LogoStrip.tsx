import { motion } from "framer-motion";

const ITEMS = [
  "datasets",
  "HuggingFace",
  "MLX",
  "Soup",
  "Apple Silicon",
  "PyTorch",
  "GGUF",
  "mlx_lm",
];

export function LogoStrip() {
  return (
    <section className="relative border-y border-white/[0.04] py-10">
      <div className="container-page">
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, amount: 0.5 }}
          transition={{ duration: 0.6 }}
          className="mb-6 text-center font-mono text-[10.5px] uppercase tracking-[0.22em] text-ink-400"
        >
          Plays nicely with the Apple Silicon stack
        </motion.p>
        <div className="overflow-hidden">
          <div className="flex w-max animate-marquee items-center gap-12 [mask-image:linear-gradient(to_right,transparent,#000_8%,#000_92%,transparent)]">
            {[...ITEMS, ...ITEMS].map((label, i) => (
              <div
                key={`${label}-${i}`}
                className="flex shrink-0 items-center gap-2 font-display text-[15px] font-medium tracking-tight text-ink-300/70 transition-colors hover:text-ink-100"
              >
                <span className="h-1.5 w-1.5 rounded-full bg-accent-400/70" />
                {label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
