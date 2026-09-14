import { Github, BookOpen, FileText, ShieldCheck, History, Heart } from "lucide-react";
import { Logo } from "./Logo";

const COLUMNS = [
  {
    title: "Product",
    links: [
      { label: "Overview", href: "#product" },
      { label: "Features", href: "#features" },
      { label: "Pipeline", href: "#pipeline" },
      { label: "Benchmarks", href: "#benchmarks" },
      { label: "Install", href: "#install" },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "README", href: "https://github.com/sachncs/hone#readme", external: true },
      { label: "Quick Start", href: "https://github.com/sachncs/hone#quick-start", external: true },
      { label: "SOTA expectations", href: "https://github.com/sachncs/hone/blob/main/docs/SOTA-EXPECTATIONS.md", external: true },
      { label: "Ablation report", href: "https://github.com/sachncs/hone/blob/main/docs/ABLATION-REPORT.md", external: true },
      { label: "Archive", href: "https://github.com/sachncs/hone/blob/main/docs/ARCHIVE.md", external: true },
    ],
  },
  {
    title: "Project",
    links: [
      { label: "GitHub", href: "https://github.com/sachncs/hone", external: true },
      { label: "Issues", href: "https://github.com/sachncs/hone/issues", external: true },
      { label: "Changelog", href: "https://github.com/sachncs/hone/blob/main/CHANGELOG.md", external: true },
      { label: "Contributing", href: "https://github.com/sachncs/hone/blob/main/CONTRIBUTING.md", external: true },
      { label: "Security", href: "https://github.com/sachncs/hone/blob/main/SECURITY.md", external: true },
    ],
  },
];

export function Footer() {
  return (
    <footer className="relative border-t border-white/[0.06]">
      <div className="container-page py-16">
        <div className="grid grid-cols-1 gap-12 lg:grid-cols-[1.4fr_2fr]">
          <div>
            <Logo />
            <p className="mt-5 max-w-sm text-[14px] leading-relaxed text-ink-300">
              Soup-ready JSONL data preparation for Apple Silicon. The precision layer between
              HuggingFace and a trained adapter.
            </p>

            <div className="mt-8 flex items-center gap-2">
              <a
                href="https://github.com/sachncs/hone"
                target="_blank"
                rel="noreferrer"
                aria-label="GitHub"
                className="grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/[0.03] text-ink-200 transition-colors hover:bg-white/[0.06] hover:text-ink-50"
              >
                <Github className="h-4 w-4" strokeWidth={1.7} />
              </a>
              <a
                href="https://github.com/sachncs/hone/blob/main/README.md"
                target="_blank"
                rel="noreferrer"
                aria-label="Read the docs"
                className="grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/[0.03] text-ink-200 transition-colors hover:bg-white/[0.06] hover:text-ink-50"
              >
                <BookOpen className="h-4 w-4" strokeWidth={1.7} />
              </a>
              <a
                href="https://github.com/sachncs/hone/blob/main/CHANGELOG.md"
                target="_blank"
                rel="noreferrer"
                aria-label="Changelog"
                className="grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/[0.03] text-ink-200 transition-colors hover:bg-white/[0.06] hover:text-ink-50"
              >
                <History className="h-4 w-4" strokeWidth={1.7} />
              </a>
              <a
                href="https://github.com/sachncs/hone/blob/main/SECURITY.md"
                target="_blank"
                rel="noreferrer"
                aria-label="Security policy"
                className="grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/[0.03] text-ink-200 transition-colors hover:bg-white/[0.06] hover:text-ink-50"
              >
                <ShieldCheck className="h-4 w-4" strokeWidth={1.7} />
              </a>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-10 sm:grid-cols-3">
            {COLUMNS.map((col) => (
              <div key={col.title}>
                <div className="font-mono text-[10.5px] uppercase tracking-[0.22em] text-ink-500">
                  {col.title}
                </div>
                <ul className="mt-5 space-y-3">
                  {col.links.map((l) => (
                    <li key={l.label}>
                      <a
                        href={l.href}
                        target={(l as { external?: boolean }).external ? "_blank" : undefined}
                        rel={(l as { external?: boolean }).external ? "noreferrer" : undefined}
                        className="text-[13.5px] text-ink-200 transition-colors hover:text-ink-50"
                      >
                        {l.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-14 flex flex-col items-start justify-between gap-4 border-t border-white/[0.06] pt-8 sm:flex-row sm:items-center">
          <div className="flex items-center gap-2 text-[12.5px] text-ink-400">
            <span>© 2026 Sachin · MIT licensed.</span>
            <span className="hidden h-3 w-px bg-white/15 sm:inline-block" />
            <span className="hidden sm:inline-flex items-center gap-1.5">
              Built with <Heart className="h-3 w-3 text-rose-400/80" strokeWidth={1.7} /> on a MacBook.
            </span>
          </div>
          <div className="flex items-center gap-4 text-[12px] text-ink-400">
            <span className="flex items-center gap-1.5">
              <FileText className="h-3.5 w-3.5" strokeWidth={1.7} />
              MIT License
            </span>
            <span>v0.3.0 · main</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
