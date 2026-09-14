import { motion, useScroll, useTransform } from "framer-motion";
import { Menu, X, Github, BookOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { Logo } from "./Logo";
import { cn } from "../lib/utils";

const NAV_LINKS = [
  { label: "Product", href: "#product" },
  { label: "Features", href: "#features" },
  { label: "Pipeline", href: "#pipeline" },
  { label: "Benchmarks", href: "#benchmarks" },
  { label: "Docs", href: "#docs" },
];

export function Nav() {
  const { scrollY } = useScroll();
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    return scrollY.on("change", (v) => setScrolled(v > 12));
  }, [scrollY]);

  useEffect(() => {
    if (open) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  const navBg = useTransform(
    scrollY,
    [0, 80],
    ["rgba(4,6,12,0)", "rgba(4,6,12,0.72)"],
  );
  const navBorder = useTransform(
    scrollY,
    [0, 80],
    ["rgba(255,255,255,0)", "rgba(255,255,255,0.06)"],
  );

  return (
    <>
      <motion.header
        style={{ background: navBg }}
        className={cn(
          "fixed inset-x-0 top-0 z-50 border-b backdrop-blur-md transition-[border-color] duration-300",
        )}
      >
        <motion.div style={{ borderColor: navBorder }} className="border-b">
          <div className="container-page flex h-[60px] items-center justify-between">
            <a href="#top" className="flex items-center gap-2.5" aria-label="hone home">
              <Logo />
              <span className="ml-1 hidden font-mono text-[10px] uppercase tracking-[0.18em] text-ink-400 sm:inline">
                v0.3.0
              </span>
            </a>

            <nav className="hidden items-center gap-1 md:flex">
              {NAV_LINKS.map((l) => (
                <a
                  key={l.href}
                  href={l.href}
                  className="rounded-full px-3.5 py-1.5 text-[13.5px] text-ink-200 transition-colors duration-200 hover:bg-white/[0.04] hover:text-ink-50"
                >
                  {l.label}
                </a>
              ))}
            </nav>

            <div className="hidden items-center gap-2 md:flex">
              <a
                href="https://github.com/sachncs/hone"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-[13px] text-ink-100 transition-all duration-200 hover:bg-white/[0.06] hover:text-ink-50"
              >
                <Github className="h-3.5 w-3.5" strokeWidth={1.8} />
                <span>Star</span>
                <span className="font-mono text-[11px] text-ink-300">·</span>
                <span className="font-mono text-[11px] text-ink-300">GitHub</span>
              </a>
              <a href="#install" className="btn-primary text-[13px]">
                <BookOpen className="h-3.5 w-3.5" strokeWidth={2} />
                <span>Get started</span>
              </a>
            </div>

            <button
              className="grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/[0.03] md:hidden"
              onClick={() => setOpen(true)}
              aria-label="Open menu"
            >
              <Menu className="h-4 w-4 text-ink-50" />
            </button>
          </div>
        </motion.div>
      </motion.header>

      {scrolled && <div className="h-[60px]" />}

      {/* Mobile drawer */}
      <div
        className={cn(
          "fixed inset-0 z-[60] md:hidden",
          open ? "pointer-events-auto" : "pointer-events-none",
        )}
        aria-hidden={!open}
      >
        <div
          className={cn(
            "absolute inset-0 bg-ink-975/80 backdrop-blur-md transition-opacity duration-300",
            open ? "opacity-100" : "opacity-0",
          )}
          onClick={() => setOpen(false)}
        />
        <div
          className={cn(
            "absolute right-0 top-0 h-full w-[78%] max-w-sm border-l border-white/10 bg-ink-950 p-6 transition-transform duration-300 ease-smooth",
            open ? "translate-x-0" : "translate-x-full",
          )}
        >
          <div className="mb-8 flex items-center justify-between">
            <Logo />
            <button
              className="grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/[0.03]"
              onClick={() => setOpen(false)}
              aria-label="Close menu"
            >
              <X className="h-4 w-4 text-ink-50" />
            </button>
          </div>
          <div className="flex flex-col gap-1">
            {NAV_LINKS.map((l) => (
              <a
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="rounded-xl px-3 py-3 text-base text-ink-100 transition-colors hover:bg-white/[0.04] hover:text-ink-50"
              >
                {l.label}
              </a>
            ))}
          </div>
          <div className="mt-8 flex flex-col gap-3">
            <a href="#install" onClick={() => setOpen(false)} className="btn-primary w-full justify-center">
              Get started
            </a>
            <a
              href="https://github.com/sachncs/hone"
              target="_blank"
              rel="noreferrer"
              onClick={() => setOpen(false)}
              className="btn-secondary w-full justify-center"
            >
              <Github className="h-4 w-4" strokeWidth={1.8} /> View on GitHub
            </a>
          </div>
        </div>
      </div>
    </>
  );
}
