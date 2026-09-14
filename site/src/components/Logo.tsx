import { cn } from "../lib/utils";

interface LogoProps {
  className?: string;
  size?: number;
  showWordmark?: boolean;
}

export function Logo({ className, size = 28, showWordmark = true }: LogoProps) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <span
        aria-hidden
        className="relative inline-flex items-center justify-center"
        style={{ width: size, height: size }}
      >
        <svg
          width={size}
          height={size}
          viewBox="0 0 32 32"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="drop-shadow-[0_0_8px_rgba(58,98,255,0.35)]"
        >
          <defs>
            <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#5e85ff" />
              <stop offset="0.5" stopColor="#8aa8ff" />
              <stop offset="1" stopColor="#c4b6ff" />
            </linearGradient>
            <linearGradient id="logo-fill" x1="6" y1="6" x2="26" y2="26" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#0d1124" />
              <stop offset="1" stopColor="#04060c" />
            </linearGradient>
          </defs>
          <rect x="1" y="1" width="30" height="30" rx="8" fill="url(#logo-fill)" stroke="url(#logo-grad)" strokeWidth="1.2" />
          <path
            d="M9 22 L16 8 L23 22 M11.5 17 H20.5"
            stroke="url(#logo-grad)"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />
          <circle cx="22.5" cy="9.5" r="1.2" fill="#8aa8ff" />
        </svg>
      </span>
      {showWordmark && (
        <span className="select-none font-display text-[15px] font-semibold tracking-tightest text-ink-50">
          hone
        </span>
      )}
    </div>
  );
}
