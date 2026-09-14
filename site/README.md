# hone site

Premium product landing page for [hone](https://github.com/sachncs/hone) —
the Soup-driven JSONL preparation library for Apple Silicon.

Built with **Vite + React + TypeScript + Tailwind CSS + Framer Motion**.

## Develop

```bash
cd site
npm install
npm run dev          # http://localhost:5173
```

## Build

```bash
npm run build        # outputs site/dist
npm run preview      # serves site/dist on /hone/
```

## Deploy

Pushing to `main` triggers `.github/workflows/pages.yml`, which builds the
site and publishes it to GitHub Pages at
`https://sachncs.github.io/hone/`.

## Structure

```
site/
├── public/             # static assets (favicon, logo, OG image)
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── components/     # Nav, Hero, Overview, Features, ...
│   ├── lib/
│   └── styles/globals.css
├── index.html
├── tailwind.config.js
├── vite.config.ts
└── package.json
```

The site is fully self-contained — no `hone` Python package is imported at
build time. All copy is hand-authored for a marketing surface.
