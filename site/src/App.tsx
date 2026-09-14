import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { LogoStrip } from "./components/LogoStrip";
import { Overview } from "./components/Overview";
import { Features } from "./components/Features";
import { Pipeline } from "./components/Pipeline";
import { Benchmarks } from "./components/Benchmarks";
import { CodePreview } from "./components/CodePreview";
import { CTA } from "./components/CTA";
import { Footer } from "./components/Footer";

export function App() {
  return (
    <div className="relative isolate min-h-screen overflow-x-hidden">
      <Nav />
      <main>
        <Hero />
        <LogoStrip />
        <Overview />
        <Features />
        <Pipeline />
        <Benchmarks />
        <CodePreview />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}
