/** @type {import("tailwindcss").Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f6f7f9",
          100: "#eceef2",
          200: "#d5dae3",
          300: "#aeb6c5",
          400: "#828ea3",
          500: "#5e6983",
          600: "#3f485d",
          700: "#2c3344",
          800: "#1a1f2c",
          900: "#0f121b",
          950: "#070912",
          975: "#04060c",
        },
        accent: {
          50: "#eef2ff",
          100: "#dde6ff",
          200: "#b8cbff",
          300: "#8aa8ff",
          400: "#5e85ff",
          500: "#3a62ff",
          600: "#2845e6",
          700: "#1f36b3",
          800: "#1c2f8a",
          900: "#1a2a6e",
        },
        steel: {
          50: "#f3f6f8",
          100: "#e1e8ee",
          200: "#c1ccd6",
          300: "#94a5b4",
          400: "#6c7f92",
          500: "#506277",
          600: "#3d4b5d",
          700: "#2d3849",
          800: "#1f2837",
          900: "#161c28",
        },
      },
      fontFamily: {
        sans: [
          "InterVariable",
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Text",
          "SF Pro Display",
          "Segoe UI",
          "Roboto",
          "system-ui",
          "sans-serif",
        ],
        mono: [
          "JetBrainsMonoVariable",
          "JetBrains Mono",
          "SF Mono",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
        display: [
          "InterDisplay",
          "InterVariable",
          "Inter",
          "-apple-system",
          "system-ui",
          "sans-serif",
        ],
      },
      letterSpacing: {
        tightest: "-0.04em",
        tighter: "-0.025em",
        tight: "-0.015em",
        snug: "-0.01em",
      },
      fontSize: {
        "display-2xl": [
          "clamp(3.5rem, 7vw, 6rem)",
          { lineHeight: "0.95", letterSpacing: "-0.04em", fontWeight: "600" },
        ],
        "display-xl": [
          "clamp(2.75rem, 5.4vw, 4.5rem)",
          { lineHeight: "1.02", letterSpacing: "-0.035em", fontWeight: "600" },
        ],
        "display-lg": [
          "clamp(2.25rem, 4.2vw, 3.5rem)",
          { lineHeight: "1.05", letterSpacing: "-0.03em", fontWeight: "600" },
        ],
        "display-md": [
          "clamp(1.75rem, 3vw, 2.5rem)",
          { lineHeight: "1.1", letterSpacing: "-0.025em", fontWeight: "600" },
        ],
      },
      boxShadow: {
        soft: "0 1px 1px rgba(0,0,0,0.04), 0 2px 4px rgba(0,0,0,0.06)",
        glow: "0 0 0 1px rgba(58,98,255,0.18), 0 10px 40px -10px rgba(58,98,255,0.35)",
        ring: "0 0 0 1px rgba(255,255,255,0.06)",
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 1px 2px rgba(0,0,0,0.2), 0 24px 60px -24px rgba(0,0,0,0.6)",
        "card-lg":
          "0 1px 0 rgba(255,255,255,0.05) inset, 0 1px 2px rgba(0,0,0,0.3), 0 40px 100px -30px rgba(0,0,0,0.7)",
      },
      backgroundImage: {
        "grid-fade":
          "linear-gradient(to bottom, transparent 0%, rgba(7,9,18,0.4) 80%, #04060c 100%)",
        "mesh-hero":
          "radial-gradient(at 12% 10%, rgba(58,98,255,0.18) 0px, transparent 55%), radial-gradient(at 88% 6%, rgba(160,108,255,0.14) 0px, transparent 55%), radial-gradient(at 50% 110%, rgba(58,98,255,0.12) 0px, transparent 60%)",
        "mesh-section":
          "radial-gradient(at 80% 0%, rgba(58,98,255,0.08) 0px, transparent 55%), radial-gradient(at 0% 100%, rgba(58,98,255,0.06) 0px, transparent 55%)",
        "noise":
          "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.06 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "0% 50%" },
          "100%": { backgroundPosition: "200% 50%" },
        },
        floaty: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
        pulse_soft: {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "1" },
        },
        marquee: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "blink-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
        },
      },
      animation: {
        shimmer: "shimmer 6s linear infinite",
        floaty: "floaty 6s ease-in-out infinite",
        "pulse-soft": "pulse_soft 3.2s ease-in-out infinite",
        marquee: "marquee 60s linear infinite",
        "fade-up": "fade-up 0.6s ease-out both",
        "blink-soft": "blink-soft 1.4s ease-in-out infinite",
      },
      transitionTimingFunction: {
        smooth: "cubic-bezier(0.22, 1, 0.36, 1)",
        spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
      },
    },
  },
  plugins: [],
};
