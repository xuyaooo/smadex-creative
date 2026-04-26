import { Link } from "react-router-dom";
import {
  motion, AnimatePresence, useScroll, useTransform, useInView, useMotionValue, useSpring,
  useMotionValueEvent, MotionValue,
} from "framer-motion";
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight, Sparkles, TrendingDown, Wand2,
  Target, ScanSearch, BarChart3, Zap, CheckCircle2, AlertTriangle, PencilLine,
  ImageIcon, X, Lightbulb,
} from "lucide-react";
import Logo from "../components/Logo";

/**
 * Apple-style continuous-scroll homepage:
 *   1. Hero (1 viewport)               — kinetic typography, magnetic CTA
 *   2. Cinematic story (4 viewports)   — sticky stage, one scrollYProgress
 *                                        drives all layers crossfading
 *                                        from "Problem" → "Solution" → "Product"
 *   3. Features marquee (1 viewport)   — horizontal pin-scroll row
 *   4. How it works (1 viewport)       — three-step handoff with bridge line
 *   5. Final CTA
 */
export default function HomePage() {
  return (
    <main className="[overflow-x:clip] bg-ink-950">
      <PageProgressRail />
      <Hero />
      <CinematicStory />
      <FeaturesMarquee />
      <HowItWorks />
      <FinalCTA />
    </main>
  );
}

/** Right-edge scroll progress rail. Fills as the user reads down. */
function PageProgressRail() {
  const { scrollYProgress } = useScroll();
  const smooth = useSpring(scrollYProgress, { stiffness: 100, damping: 25, mass: 0.4 });
  return (
    <motion.div
      aria-hidden
      style={{ scaleY: smooth, transformOrigin: "top" }}
      className="pointer-events-none fixed right-0 top-0 z-40 h-screen w-[2px] bg-gradient-to-b from-brand-500 via-pink-500 to-cyan-400"
    />
  );
}

/* ============================================================
   1. HERO
   ============================================================ */
type Sample = {
  id: number;
  score: number;
  action: "Scale" | "Watch" | "Maintain" | "Pause/Pivot";
  color: "emerald" | "amber" | "sky" | "rose";
  status: string;
  vertical: string;
  format: string;
  probabilities: { top: number; stable: number; fatigued: number; under: number };
  strengths: string[];
  weaknesses: string[];
  recommendation: string;
};

const HERO_SAMPLES: Sample[] = [
  {
    id: 500003, score: 87, action: "Scale", color: "emerald",
    status: "Top Performer", vertical: "Gaming", format: "rewarded video",
    probabilities: { top: 0.71, stable: 0.21, fatigued: 0.06, under: 0.02 },
    strengths: ["Strong CTA contrast", "Brand visible <2s", "Vivid color palette"],
    weaknesses: ["Text density slightly above benchmark"],
    recommendation: "Push more spend now while CTR is climbing.",
  },
  {
    id: 500144, score: 38, action: "Watch", color: "amber",
    status: "Fatigued", vertical: "Gaming", format: "interstitial",
    probabilities: { top: 0.00, stable: 0.27, fatigued: 0.72, under: 0.01 },
    strengths: ["Clear value proposition headline"],
    weaknesses: ["No visible product shot", "CTA blends into background", "CTR collapsed in last 7d"],
    recommendation: "Refresh creative or rotate to a new variant within 48h.",
  },
  {
    id: 500052, score: 71, action: "Maintain", color: "sky",
    status: "Stable", vertical: "Travel", format: "rewarded video",
    probabilities: { top: 0.18, stable: 0.74, fatigued: 0.07, under: 0.01 },
    strengths: ["Photo-real scene with strong emotional tone", "Brand visible end-card"],
    weaknesses: ["Mid-tier hook clarity", "Limited urgency signaling"],
    recommendation: "Keep running. Test a discount-badge variant for upside.",
  },
];

function Hero() {
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const yBg     = useTransform(scrollYProgress, [0, 1], [0, 220]);
  const opacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const titleY  = useTransform(scrollYProgress, [0, 1], [0, -80]);
  const titleS  = useTransform(scrollYProgress, [0, 1], [1, 0.92]);

  return (
    <section ref={ref} className="relative min-h-screen flex items-center pt-24 pb-12">
      <motion.div style={{ y: yBg, opacity }} className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-20 h-[34rem] w-[34rem] rounded-full bg-brand-500/35 blur-3xl animate-blob" />
        <div className="absolute top-40 -right-24 h-[30rem] w-[30rem] rounded-full bg-pink-500/30 blur-3xl animate-blob [animation-delay:-6s]" />
        <div className="absolute bottom-0 left-1/3 h-[24rem] w-[24rem] rounded-full bg-cyan-400/20 blur-3xl animate-blob [animation-delay:-12s]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.04)_1px,transparent_0)] [background-size:32px_32px] mask-fade" />
      </motion.div>

      <Particles />

      <div className="container-narrow relative">
        <motion.div style={{ y: titleY, scale: titleS }} className="flex flex-col items-center text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.8, rotate: -8 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            transition={{ duration: 0.7, delay: 0.1, type: "spring", bounce: 0.4 }}
            className="mt-4"
          >
            <Logo size={88} className="drop-shadow-[0_15px_40px_rgba(99,102,241,0.6)]" />
          </motion.div>

          <h1 className="mt-7 font-display text-3xl font-extrabold tracking-tight leading-[1] sm:text-5xl md:text-6xl lg:text-7xl">
            <WordReveal text="Do your ad campaigns" delay={0.2} className="text-white" />
            <br />
            <KineticGradient>
              <WordReveal text="not reach what you expected?" delay={0.55} />
            </KineticGradient>
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.32 }}
            className="mt-7 max-w-2xl text-balance text-base text-slate-300 sm:text-lg leading-relaxed"
          >
            Meet your <span className="font-semibold text-white">AI Creative Companion</span>. The co pilot that scores, explains and rewrites your creatives in seconds.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.45 }}
            className="mt-10 flex flex-col sm:flex-row items-center gap-3"
          >
            <MagneticLink to="/predict" className="btn-primary group text-base px-7 py-3 shadow-glow">
              <Zap className="h-4 w-4" />
              Start now
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </MagneticLink>
            <a href="#story" className="btn-ghost">See how it works</a>
          </motion.div>

          {/* Live preview strip: click any sample to open the popup */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.65, duration: 0.7 }}
            className="mt-12 flex items-center justify-center gap-3"
          >
            {HERO_SAMPLES.map((c) => (
              <motion.button
                key={c.id}
                onClick={() => setSelectedSample(c)}
                whileHover={{ y: -8, scale: 1.05 }}
                whileTap={{ scale: 0.97 }}
                transition={{ type: "spring", stiffness: 300, damping: 22 }}
                className="group relative w-24 sm:w-28 rounded-xl overflow-hidden border border-white/15 bg-black/30 shadow-[0_20px_60px_-20px_rgba(99,102,241,0.5)] cursor-pointer text-left"
              >
                <div className="aspect-square">
                  <img
                    src={`/assets/creative_${c.id}.png`}
                    alt={`creative ${c.id}`}
                    className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-110"
                    onError={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "0"; }}
                  />
                </div>
                <div className="absolute inset-x-0 bottom-0 p-1.5 bg-gradient-to-t from-black/95 to-transparent">
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono tabular-nums text-base font-extrabold text-white leading-none">{c.score}</span>
                    <span className={`text-[8px] font-bold uppercase tracking-wider rounded px-1 py-0.5
                      ${c.color === "emerald" ? "text-emerald-200 bg-emerald-500/30"
                        : c.color === "amber" ? "text-amber-200 bg-amber-500/30"
                        : "text-sky-200 bg-sky-500/30"}`}>
                      {c.action}
                    </span>
                  </div>
                </div>
                {/* Subtle "click me" pulse ring */}
                <span className="absolute inset-0 rounded-xl ring-2 ring-brand-500/0 group-hover:ring-brand-500/40 transition" />
              </motion.button>
            ))}
          </motion.div>

          <p className="mt-3 text-[11px] text-slate-500 uppercase tracking-[0.25em]">tap a sample to see its breakdown</p>

          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.85, duration: 0.7 }}
            className="mt-6 flex flex-wrap justify-center items-center gap-x-6 gap-y-2 text-[11px] uppercase tracking-[0.25em] text-slate-500"
          >
            <span className="flex items-center gap-2"><CheckCircle2 className="h-3 w-3 text-emerald-400" /> 1,076 creatives benchmarked</span>
            <span className="flex items-center gap-2"><CheckCircle2 className="h-3 w-3 text-emerald-400" /> 6 model ensemble</span>
            <span className="flex items-center gap-2"><CheckCircle2 className="h-3 w-3 text-emerald-400" /> AUC 0.94</span>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1, duration: 0.6 }}
            className="mt-12 flex flex-col items-center gap-2 text-slate-400 text-xs"
          >
            <motion.div
              animate={{ y: [0, 10, 0] }} transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
              className="text-3xl" aria-hidden
            >
              👇
            </motion.div>
            <span className="uppercase tracking-[0.3em]">scroll</span>
          </motion.div>
        </motion.div>
      </div>

      <style>{`
        .mask-fade {
          mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
          -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
        }
      `}</style>

      <AnimatePresence>
        {selectedSample && (
          <SampleModal sample={selectedSample} onClose={() => setSelectedSample(null)} />
        )}
      </AnimatePresence>
    </section>
  );
}

function SampleModal({ sample, onClose }: { sample: Sample; onClose: () => void }) {
  // Close on Escape + scroll lock that PRESERVES the page's scroll position.
  // Naive `overflow: hidden` causes the page to jump to top when restored —
  // we pin body to the current scroll Y, then restore on close.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);

    const scrollY = window.scrollY;
    const scrollbar = window.innerWidth - document.documentElement.clientWidth;
    document.body.style.position = "fixed";
    document.body.style.top = `-${scrollY}px`;
    document.body.style.left = "0";
    document.body.style.right = "0";
    document.body.style.width = "100%";
    if (scrollbar > 0) document.body.style.paddingRight = `${scrollbar}px`;

    return () => {
      window.removeEventListener("keydown", handler);
      document.body.style.position = "";
      document.body.style.top = "";
      document.body.style.left = "";
      document.body.style.right = "";
      document.body.style.width = "";
      document.body.style.paddingRight = "";
      window.scrollTo(0, scrollY);
    };
  }, [onClose]);

  const tone = sample.color === "emerald" ? "text-emerald-300 bg-emerald-500/20 border-emerald-500/40"
             : sample.color === "amber" ? "text-amber-300 bg-amber-500/20 border-amber-500/40"
             : sample.color === "sky" ? "text-sky-300 bg-sky-500/20 border-sky-500/40"
             : "text-rose-300 bg-rose-500/20 border-rose-500/40";

  // Stagger config for child animations
  const container = {
    hidden: {},
    show: { transition: { staggerChildren: 0.06, delayChildren: 0.15 } },
  };
  const item = {
    hidden: { opacity: 0, y: 12 },
    show:   { opacity: 1, y: 0, transition: { type: "spring", stiffness: 260, damping: 22 } },
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onClose}
      className="fixed inset-0 z-[60] grid place-items-center bg-ink-950/85 backdrop-blur-md p-4"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.92, y: 20 }}
        animate={{ opacity: 1, scale: 1,    y: 0  }}
        exit={{    opacity: 0, scale: 0.95, y: 10 }}
        transition={{ type: "spring", stiffness: 280, damping: 28 }}
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-5xl max-h-[92vh] grid grid-cols-1 md:grid-cols-[1.05fr_1.4fr] rounded-3xl border border-white/15 bg-ink-900/95 backdrop-blur-xl overflow-hidden shadow-[0_50px_140px_-20px_rgba(99,102,241,0.55)]"
      >
        {/* atmospheric blobs */}
        <motion.div
          className="pointer-events-none absolute -top-24 -right-16 h-72 w-72 rounded-full bg-brand-500/35 blur-3xl"
          animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0.7, 0.5] }}
          transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="pointer-events-none absolute -bottom-24 -left-16 h-72 w-72 rounded-full bg-pink-500/30 blur-3xl"
          animate={{ scale: [1, 1.15, 1], opacity: [0.4, 0.65, 0.4] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />

        <button
          onClick={onClose}
          className="absolute top-3 right-3 z-20 h-8 w-8 grid place-items-center rounded-full bg-black/60 text-slate-200 hover:bg-black/90 hover:text-white transition"
          aria-label="Close"
        >
          <X className="h-3.5 w-3.5" />
        </button>

        {/* LEFT — image */}
        <div className="relative bg-black/70 min-h-[260px] md:min-h-[unset]">
          <img
            src={`/assets/creative_${sample.id}.png`}
            alt={`creative ${sample.id}`}
            className="absolute inset-0 h-full w-full object-cover"
          />
          <div className="absolute inset-x-0 top-0 p-3 flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider bg-black/70 backdrop-blur rounded px-2 py-1 text-white">
              sample #{sample.id}
            </span>
            <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full backdrop-blur border ${tone}`}>
              <Target className="inline h-3 w-3 mr-1" /> {sample.action}
            </span>
          </div>
          <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/95 to-transparent">
            <div className="text-sm font-semibold text-white capitalize">{sample.vertical}</div>
            <div className="text-[10px] text-white/70 uppercase tracking-wider">{sample.format}</div>
          </div>
        </div>

        {/* RIGHT — info, fits viewport without scroll on most laptops */}
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="relative p-5 sm:p-6 flex flex-col gap-4 min-h-0"
        >
          {/* Health hero */}
          <motion.div
            variants={item}
            className="rounded-xl bg-gradient-to-br from-brand-500/20 to-pink-500/10 border border-white/10 p-4 relative overflow-hidden flex-shrink-0"
          >
            <div className="absolute -top-12 -right-8 h-32 w-32 rounded-full bg-brand-500/30 blur-2xl pointer-events-none" />
            <div className="relative flex items-center justify-between gap-3">
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-300">creative health score</div>
                <div className="mt-0.5 flex items-baseline gap-1">
                  <CountingNumber to={sample.score} className="font-display text-5xl font-extrabold tabular-nums leading-none" />
                  <span className="text-slate-400 text-sm">/100</span>
                </div>
              </div>
              <ListBadge label="Predicted" value={sample.status} />
            </div>
            <div className="mt-3 h-1.5 rounded-full bg-white/10 overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-brand-500 to-pink-500"
                initial={{ width: 0 }} animate={{ width: `${sample.score}%` }}
                transition={{ duration: 1, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
          </motion.div>

          {/* Probabilities */}
          <motion.div variants={item} className="flex-shrink-0">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1.5">class probabilities</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
              {[
                { l: "top performer",  v: sample.probabilities.top,      c: "emerald" },
                { l: "stable",         v: sample.probabilities.stable,   c: "sky" },
                { l: "fatigued",       v: sample.probabilities.fatigued, c: "amber" },
                { l: "underperformer", v: sample.probabilities.under,    c: "rose" },
              ].map((row, i) => (
                <div key={row.l} className="text-[10px]">
                  <div className="flex justify-between">
                    <span className={`capitalize ${probColor(row.c)}`}>{row.l}</span>
                    <span className="tabular-nums text-slate-400">{(row.v * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1 mt-0.5 rounded-full bg-white/5">
                    <motion.div
                      className={`h-full rounded-full ${probBar(row.c)}`}
                      initial={{ width: 0 }} animate={{ width: `${row.v * 100}%` }}
                      transition={{ duration: 0.7, delay: 0.4 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Strengths + weaknesses (side by side, compact) */}
          <motion.div variants={item} className="grid grid-cols-2 gap-2 text-[10px] min-h-0 flex-shrink-0">
            <div className="rounded-lg bg-white/[0.04] border border-emerald-500/20 p-2.5">
              <div className="flex items-center gap-1 text-emerald-300 uppercase tracking-wider font-bold mb-1">
                <CheckCircle2 className="h-3 w-3" /> strengths
              </div>
              <ul className="space-y-0.5 text-slate-200 leading-tight">
                {sample.strengths.map((s) => (
                  <li key={s} className="flex gap-1"><span className="text-emerald-400">✓</span><span>{s}</span></li>
                ))}
              </ul>
            </div>
            <div className="rounded-lg bg-white/[0.04] border border-rose-500/20 p-2.5">
              <div className="flex items-center gap-1 text-rose-300 uppercase tracking-wider font-bold mb-1">
                <AlertTriangle className="h-3 w-3" /> weaknesses
              </div>
              <ul className="space-y-0.5 text-slate-200 leading-tight">
                {sample.weaknesses.map((s) => (
                  <li key={s} className="flex gap-1"><span className="text-rose-400">✗</span><span>{s}</span></li>
                ))}
              </ul>
            </div>
          </motion.div>

          {/* Top recommendation */}
          <motion.div
            variants={item}
            className="rounded-xl bg-gradient-to-br from-brand-500/20 to-pink-500/10 border border-brand-500/30 p-3.5 flex-shrink-0"
          >
            <div className="flex items-center gap-1 text-brand-200 text-[10px] uppercase tracking-wider font-bold mb-1">
              <Lightbulb className="h-3 w-3" /> top recommendation
            </div>
            <p className="text-sm text-white font-medium leading-snug">{sample.recommendation}</p>
          </motion.div>

          {/* CTA buttons */}
          <motion.div variants={item} className="mt-auto flex gap-2 pt-1">
            <Link to="/predict" className="btn-primary flex-1 justify-center group text-sm py-2">
              <Zap className="h-4 w-4" /> Try with your ad
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
            <button onClick={onClose} className="btn-ghost text-sm py-2 px-4">Close</button>
          </motion.div>
        </motion.div>
      </motion.div>
    </motion.div>
  );
}

function ListBadge({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider rounded-full bg-white/10 border border-white/15 px-2 py-0.5">
      <span className="text-slate-400">{label}</span>
      <span className="text-white">{value}</span>
    </span>
  );
}

function probColor(c: string) {
  return c === "emerald" ? "text-emerald-300"
       : c === "sky" ? "text-sky-300"
       : c === "amber" ? "text-amber-300"
       : "text-rose-300";
}
function probBar(c: string) {
  return c === "emerald" ? "bg-emerald-500/70"
       : c === "sky" ? "bg-sky-500/70"
       : c === "amber" ? "bg-amber-500/70"
       : "bg-rose-500/70";
}

function WordReveal({ text, delay = 0, className = "" }: { text: string; delay?: number; className?: string }) {
  const words = text.split(" ");
  return (
    <span className={`inline ${className}`}>
      {words.map((w, i) => (
        <span key={`${w}-${i}`} className="inline-block overflow-hidden align-bottom">
          <motion.span
            initial={{ y: "110%" }}
            animate={{ y: "0%" }}
            transition={{ duration: 0.7, delay: delay + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
            className="inline-block"
          >
            {w}{i < words.length - 1 ? " " : ""}
          </motion.span>
        </span>
      ))}
    </span>
  );
}

function KineticGradient({ children }: { children: React.ReactNode }) {
  return (
    <span className="relative inline-block">
      <span className="bg-gradient-to-r from-brand-300 via-pink-300 to-amber-200 bg-clip-text text-transparent animate-gradient-pan bg-[length:200%_auto]">
        {children}
      </span>
      <style>{`
        @keyframes gradient-pan { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        .animate-gradient-pan { animation: gradient-pan 8s ease-in-out infinite; }
      `}</style>
    </span>
  );
}

function MagneticLink({ to, children, className = "" }: { to: string; children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLAnchorElement>(null);
  const x = useMotionValue(0); const y = useMotionValue(0);
  const sx = useSpring(x, { stiffness: 300, damping: 20 });
  const sy = useSpring(y, { stiffness: 300, damping: 20 });

  return (
    <Link
      to={to} ref={ref as any} className={className}
      onMouseMove={(e) => {
        const r = ref.current?.getBoundingClientRect(); if (!r) return;
        x.set(((e.clientX - r.left) - r.width / 2) * 0.25);
        y.set(((e.clientY - r.top) - r.height / 2) * 0.25);
      }}
      onMouseLeave={() => { x.set(0); y.set(0); }}
    >
      <motion.span style={{ x: sx, y: sy }} className="inline-flex items-center gap-2">
        {children}
      </motion.span>
    </Link>
  );
}

function Particles() {
  // Fewer particles, slower fade so the page feels calmer.
  const dots = Array.from({ length: 12 }).map((_, i) => ({
    id: i,
    left: `${(i * 73) % 100}%`,
    top: `${(i * 41) % 100}%`,
    size: 1 + (i % 3),
    delay: (i % 6) * 0.8,
  }));
  return (
    <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
      {dots.map((d) => (
        <motion.span
          key={d.id} className="absolute rounded-full bg-white/30"
          style={{ left: d.left, top: d.top, width: d.size, height: d.size }}
          animate={{ opacity: [0.05, 0.4, 0.05] }}
          transition={{ duration: 8 + (d.id % 4), repeat: Infinity, delay: d.delay, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
}

/* ============================================================
   2. CINEMATIC STORY
   One sticky stage, one scrollYProgress, all layers crossfade.
   - 0.00 → 0.25 : "Your ads are dying"  (red, decline)
   - 0.25 → 0.50 : "How much are you losing?"  (numbers ramp)
   - 0.50 → 0.75 : "Smadex AI enters"   (decline morphs to rise)
   - 0.75 → 1.00 : Dashboard pulls into focus
   ============================================================ */
function CinematicStory() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress: rawProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  // Smooth the raw scroll value with a spring. Higher damping + lower stiffness
  // = silkier transitions. No jitter, no fighting the user's scroll input.
  const scrollYProgress = useSpring(rawProgress, { stiffness: 60, damping: 28, mass: 0.5 });

  // Background tint shifts from rose → emerald across the whole section
  const bgRose    = useTransform(scrollYProgress, [0, 0.55], [0.45, 0]);
  const bgEmerald = useTransform(scrollYProgress, [0.4, 1], [0, 0.4]);
  const bgBrand   = useTransform(scrollYProgress, [0.55, 1], [0, 0.5]);

  // Scene timing — each scene gets a long DWELL plateau so the user can read
  // before the next scene appears. Transitions are quick (4% windows), dwell
  // takes up most of each quarter (~16% of scroll = ~75vh per scene).
  //
  //   Scene 1 dwell: 0.04 – 0.20  (transition out 0.20 – 0.24)
  //   Scene 2 dwell: 0.30 – 0.46  (transition in 0.26 – 0.30, out 0.46 – 0.50)
  //   Scene 3 dwell: 0.56 – 0.72  (transition in 0.52 – 0.56, out 0.72 – 0.76)
  //   Scene 4 dwell: 0.82 – 1.00  (transition in 0.78 – 0.82)
  const scene1Opacity = useTransform(scrollYProgress, [0.00, 0.04, 0.20, 0.24], [1, 1, 1, 0]);
  const scene2Opacity = useTransform(scrollYProgress, [0.24, 0.30, 0.46, 0.50], [0, 1, 1, 0]);
  const scene3Opacity = useTransform(scrollYProgress, [0.50, 0.56, 0.72, 0.76], [0, 1, 1, 0]);
  const scene4Opacity = useTransform(scrollYProgress, [0.76, 0.82, 1.00], [0, 1, 1]);

  // Parallax only happens during the OUT transition. Dwell is perfectly still.
  const scene1Y = useTransform(scrollYProgress, [0.20, 0.24], [0, -40]);
  const scene2Y = useTransform(scrollYProgress, [0.46, 0.50], [0, -40]);
  const scene3Y = useTransform(scrollYProgress, [0.72, 0.76], [0, -40]);
  const scene4Y = useTransform(scrollYProgress, [0.76, 0.82], [40, 0]);

  // Tiny scale settle as scenes enter/exit; dwell range is fixed at scale 1
  const scene1Scale = useTransform(scrollYProgress, [0.20, 0.24], [1, 0.97]);
  const scene2Scale = useTransform(scrollYProgress, [0.24, 0.30, 0.46, 0.50], [1.03, 1, 1, 0.97]);
  const scene3Scale = useTransform(scrollYProgress, [0.50, 0.56, 0.72, 0.76], [1.03, 1, 1, 0.97]);

  // Charts inside SCENE 3 (dwell 0.56 – 0.72): decline draws first, then
  // morphs into the rise across the dwell window so the morph is visible.
  const declineDraw = useTransform(scrollYProgress, [0.54, 0.60], [0, 1]);
  const declineFade = useTransform(scrollYProgress, [0.62, 0.66], [1, 0]);
  const riseDraw    = useTransform(scrollYProgress, [0.64, 0.72], [0, 1]);

  // Number ticker for SCENE 2 (dwell 0.30 – 0.46): values ramp early in the
  // dwell, then sit at their final value so the user can read.
  const ctrPct        = useTransform(scrollYProgress, [0.30, 0.40], [3.2, 0.8]);
  const wastedDollars = useTransform(scrollYProgress, [0.30, 0.40], [0, 18400]);

  // Continuous-rotating glow halo (lives across all scenes)
  const haloRotate = useTransform(scrollYProgress, [0, 1], [0, 220]);
  const haloScale  = useTransform(scrollYProgress, [0, 0.5, 1], [0.9, 1.15, 1.0]);
  const [ctr, setCtr] = useState("3.20");
  const [wasted, setWasted] = useState("0");
  useMotionValueEvent(ctrPct, "change", (v) => setCtr(v.toFixed(2)));
  useMotionValueEvent(wastedDollars, "change", (v) => setWasted(Math.round(v).toLocaleString()));

  // Headline morph: a single absolute element, content swaps via opacity layers below
  return (
    <section id="story" ref={ref} className="relative min-h-[480vh]">
      {/* sticky stage */}
      <div className="sticky top-0 h-screen w-full overflow-hidden">
        {/* Layer A — animated backdrop */}
        <motion.div
          className="absolute inset-0 -z-20"
          style={{
            background: "radial-gradient(ellipse 70% 50% at 50% 50%, rgba(15,23,42,0.7), rgba(11,16,32,0.95))",
          }}
        />
        <motion.div
          className="absolute inset-0 -z-10"
          style={{
            background: useTemplate(
              ["radial-gradient(60% 50% at 50% 30%, rgba(244,63,94,", ") 0%, transparent 60%)"],
              [bgRose],
            ) as any,
          }}
        />
        <motion.div
          className="absolute inset-0 -z-10"
          style={{
            background: useTemplate(
              ["radial-gradient(60% 50% at 50% 70%, rgba(52,211,153,", ") 0%, transparent 60%)"],
              [bgEmerald],
            ) as any,
          }}
        />
        <motion.div
          className="absolute inset-0 -z-10"
          style={{
            background: useTemplate(
              ["radial-gradient(70% 50% at 80% 30%, rgba(99,102,241,", ") 0%, transparent 60%)"],
              [bgBrand],
            ) as any,
          }}
        />

        {/* Persistent floating mesh dots */}
        <Particles />

        {/* Continuous rotating halo — lives across all scenes for visual flow */}
        <motion.div
          style={{ rotate: haloRotate, scale: haloScale }}
          className="pointer-events-none absolute inset-0 -z-10 grid place-items-center"
        >
          <div className="h-[60vh] w-[60vh] rounded-full opacity-40"
            style={{
              background: "conic-gradient(from 0deg, rgba(99,102,241,0.3), rgba(236,72,153,0.3), rgba(34,211,238,0.3), rgba(52,211,153,0.3), rgba(99,102,241,0.3))",
              filter: "blur(70px)",
            }}
          />
        </motion.div>

        {/* SCENE 1 — Problem reveal */}
        <motion.div
          style={{ opacity: scene1Opacity, y: scene1Y, scale: scene1Scale, pointerEvents: useTransform(scene1Opacity, (o) => o > 0.5 ? "auto" : "none") as any }}
          className="absolute inset-0 flex items-center justify-center px-6"
        >
          {/* Ghostly creatives "dying" in the backdrop. Hidden on small
              screens so they don't crowd the headline. */}
          <div className="pointer-events-none absolute inset-0 overflow-hidden hidden sm:block">
            {[
              { id: 500024, x: "6%",  y: "18%",  rotate: -8, score: 12, scale: 0.9 },
              { id: 500052, x: "82%", y: "15%",  rotate: 6,  score: 18, scale: 1.05 },
              { id: 500088, x: "10%", y: "68%",  rotate: 4,  score: 22, scale: 0.85 },
              { id: 500120, x: "80%", y: "65%",  rotate: -5, score: 8,  scale: 1.0  },
            ].map((g, i) => (
              <motion.div
                key={g.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: [0, 0.4, 0.15, 0.4, 0.1], y: [20, 0, 0, 0, -10] }}
                transition={{ duration: 6, repeat: Infinity, delay: i * 0.7, ease: "easeInOut" }}
                style={{ left: g.x, top: g.y, rotate: g.rotate, scale: g.scale }}
                className="absolute w-24 h-28 lg:w-28 lg:h-32 rounded-xl overflow-hidden border border-rose-500/30 grayscale"
              >
                <img
                  src={`/assets/creative_${g.id}.png`}
                  alt=""
                  aria-hidden
                  className="absolute inset-0 h-full w-full object-cover"
                  onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                />
                <div className="absolute inset-0 bg-rose-950/50" />
                <div className="absolute inset-x-0 bottom-0 p-1.5 bg-gradient-to-t from-rose-950 to-transparent">
                  <div className="font-mono text-base font-extrabold tabular-nums text-rose-300/80 leading-none">{g.score}</div>
                  <div className="text-[8px] uppercase tracking-wider text-rose-400/70">fatigued</div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="relative text-center max-w-3xl">
            <motion.div
              initial={{ scale: 0.9 }} whileInView={{ scale: 1 }} transition={{ duration: 0.6 }}
              className="inline-flex items-center gap-2 stat-pill"
            >
              <AlertTriangle className="h-3 w-3 text-rose-300" /> the problem
            </motion.div>
            <h2 className="mt-6 font-display text-4xl sm:text-6xl md:text-8xl font-extrabold tracking-tight leading-[0.95] text-white drop-shadow-[0_8px_30px_rgba(244,63,94,0.25)]">
              Most ads die <span className="text-rose-300">quietly.</span>
            </h2>
            <p className="mt-6 text-slate-300 text-base sm:text-lg max-w-xl mx-auto leading-relaxed px-2">
              CTR fades. ROAS slips. By the time the dashboard turns red, you've already burnt half the budget.
            </p>
          </div>
        </motion.div>

        {/* SCENE 2 — Numbers ramp */}
        <motion.div
          style={{ opacity: scene2Opacity, y: scene2Y, scale: scene2Scale, pointerEvents: useTransform(scene2Opacity, (o) => o > 0.5 ? "auto" : "none") as any }}
          className="absolute inset-0 flex items-center justify-center px-6"
        >
          <div className="text-center w-full max-w-5xl">
            <div className="inline-flex items-center gap-2 stat-pill">
              <span className="h-1.5 w-1.5 rounded-full bg-rose-400 animate-pulse" />
              <span className="text-[10px] sm:text-[11px] uppercase tracking-[0.25em] text-rose-200">your campaign right now</span>
            </div>
            <div className="mt-8 sm:mt-10 grid grid-cols-1 sm:grid-cols-2 gap-8 sm:gap-16 lg:gap-20 items-center sm:items-end">
              <div className="text-center sm:text-right">
                <div className="text-[10px] sm:text-xs text-slate-400 uppercase tracking-[0.2em] mb-2">ctr collapsing</div>
                <div className="font-display font-extrabold tabular-nums text-rose-200 leading-[0.85] text-6xl sm:text-7xl md:text-8xl lg:text-9xl drop-shadow-[0_8px_30px_rgba(244,63,94,0.5)]">
                  {ctr}<span className="text-2xl sm:text-3xl md:text-5xl text-rose-300/70 align-top">%</span>
                </div>
                <div className="mt-3 sm:mt-4 inline-flex items-center gap-2 text-[10px] sm:text-xs text-rose-200/80 bg-rose-500/10 border border-rose-500/30 rounded-full px-3 py-1">
                  <TrendingDown className="h-3 w-3" /> from 3.20% in 14 days
                </div>
              </div>
              <div className="text-center sm:text-left">
                <div className="text-[10px] sm:text-xs text-slate-400 uppercase tracking-[0.2em] mb-2">budget wasted</div>
                <div className="font-display font-extrabold tabular-nums text-amber-200 leading-[0.85] text-6xl sm:text-7xl md:text-8xl lg:text-9xl drop-shadow-[0_8px_30px_rgba(251,191,36,0.45)]">
                  <span className="text-2xl sm:text-3xl md:text-5xl text-amber-300/70 align-top">$</span>{wasted}
                </div>
                <div className="mt-3 sm:mt-4 inline-flex items-center gap-2 text-[10px] sm:text-xs text-amber-200/80 bg-amber-500/10 border border-amber-500/30 rounded-full px-3 py-1">
                  <AlertTriangle className="h-3 w-3" /> on under-performing creatives
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* SCENE 3 — Solution enters: decline morphs into rise */}
        <motion.div
          style={{ opacity: scene3Opacity, y: scene3Y, scale: scene3Scale, pointerEvents: useTransform(scene3Opacity, (o) => o > 0.5 ? "auto" : "none") as any }}
          className="absolute inset-0 flex items-center justify-center px-6"
        >
          <div className="text-center max-w-4xl w-full px-2">
            <motion.div
              initial={{ scale: 0.95 }} whileInView={{ scale: 1 }} transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 stat-pill"
            >
              <Sparkles className="h-3 w-3 text-brand-300" /> Ensemble models · Personalized AI
            </motion.div>
            <h2 className="mt-5 sm:mt-6 font-display text-3xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1]">
              <span className="text-white">Know when to </span>
              <span className="text-rose-300 drop-shadow-[0_8px_30px_rgba(244,63,94,0.4)]">stop</span>
              <span className="text-white">.</span>
              <br />
              <span className="text-white">Know when to </span>
              <span className="text-emerald-300 drop-shadow-[0_8px_30px_rgba(52,211,153,0.4)]">push.</span>
            </h2>

            {/* Morphing chart — bolder strokes, gradient area fill, axis labels */}
            <div className="mt-12 mx-auto max-w-3xl">
              <svg viewBox="0 0 100 60" className="w-full h-44 sm:h-52" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="riseGrad" x1="0" y1="0" x2="100" y2="0">
                    <stop offset="0%" stopColor="#34d399" />
                    <stop offset="100%" stopColor="#22d3ee" />
                  </linearGradient>
                  <linearGradient id="riseFill" x1="0" y1="0" x2="0" y2="60">
                    <stop offset="0%" stopColor="#34d399" stopOpacity="0.45" />
                    <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
                  </linearGradient>
                  <linearGradient id="declineFill" x1="0" y1="0" x2="0" y2="60">
                    <stop offset="0%" stopColor="#fb7185" stopOpacity="0.05" />
                    <stop offset="100%" stopColor="#fb7185" stopOpacity="0.4" />
                  </linearGradient>
                  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="0.8" />
                  </filter>
                </defs>

                {/* gridlines */}
                {[10, 25, 40, 55].map((y) => (
                  <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(255,255,255,0.07)" strokeDasharray="0.8 1.4" strokeWidth="0.2" />
                ))}

                {/* decline area fill */}
                <motion.path
                  d="M 0,15 L 14,18 L 28,22 L 42,30 L 56,38 L 70,46 L 84,52 L 100,58 L 100,60 L 0,60 Z"
                  fill="url(#declineFill)"
                  style={{ opacity: declineFade }}
                />
                {/* decline line — glow trail behind a sharper top stroke */}
                <motion.path
                  d="M 0,15 L 14,18 L 28,22 L 42,30 L 56,38 L 70,46 L 84,52 L 100,58"
                  fill="none" stroke="#fb7185" strokeWidth="1.2" strokeLinecap="round" filter="url(#glow)" opacity="0.7"
                  style={{ pathLength: declineDraw, opacity: declineFade }}
                />
                <motion.path
                  d="M 0,15 L 14,18 L 28,22 L 42,30 L 56,38 L 70,46 L 84,52 L 100,58"
                  fill="none" stroke="#fb7185" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"
                  style={{ pathLength: declineDraw, opacity: declineFade }}
                />

                {/* rise area fill */}
                <motion.path
                  d="M 0,55 L 14,48 L 28,42 L 42,32 L 56,24 L 70,14 L 84,8 L 100,4 L 100,60 L 0,60 Z"
                  fill="url(#riseFill)"
                  style={{ opacity: riseDraw }}
                />
                {/* rise line — glow + sharp top stroke */}
                <motion.path
                  d="M 0,55 L 14,48 L 28,42 L 42,32 L 56,24 L 70,14 L 84,8 L 100,4"
                  fill="none" stroke="url(#riseGrad)" strokeWidth="1.4" strokeLinecap="round" filter="url(#glow)" opacity="0.8"
                  style={{ pathLength: riseDraw }}
                />
                <motion.path
                  d="M 0,55 L 14,48 L 28,42 L 42,32 L 56,24 L 70,14 L 84,8 L 100,4"
                  fill="none" stroke="url(#riseGrad)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"
                  style={{ pathLength: riseDraw }}
                />

                {/* end-point markers + percent badges */}
                <motion.g style={{ opacity: declineFade }}>
                  <circle cx="100" cy="58" r="2.2" fill="#fb7185" stroke="#0b1020" strokeWidth="0.8" />
                </motion.g>
                <motion.g style={{ opacity: riseDraw }}>
                  <circle cx="100" cy="4" r="2.2" fill="#34d399" stroke="#0b1020" strokeWidth="0.8" />
                </motion.g>
              </svg>
              <div className="mt-2 flex justify-between text-[11px] text-slate-400 font-mono uppercase tracking-wider">
                <motion.span style={{ opacity: declineFade }} className="text-rose-300">↓ −72% CTR fatigue</motion.span>
                <motion.span style={{ opacity: riseDraw }} className="text-emerald-300">↑ +218% health</motion.span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* SCENE 4 — Dashboard pulls into focus */}
        <motion.div
          style={{ opacity: scene4Opacity, y: scene4Y, pointerEvents: useTransform(scene4Opacity, (o) => o > 0.5 ? "auto" : "none") as any }}
          className="absolute inset-0 flex flex-col items-center justify-center px-4"
        >
          <div className="text-center mb-6">
            <div className="text-[11px] uppercase tracking-[0.25em] text-brand-200/80">your ad, scored</div>
          </div>
          <DashboardFrame progress={scrollYProgress} />
        </motion.div>

        {/* Stage progress dots */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex gap-2">
          {[0, 0.25, 0.5, 0.75].map((p, i) => (
            <ProgressDot key={i} progress={scrollYProgress} threshold={p} />
          ))}
        </div>
      </div>
    </section>
  );
}

function DashboardFrame({ progress }: { progress: MotionValue<number> }) {
  // Frame zooms IN during SCENE 4's transition window (0.78 → 0.86), then
  // dwells fully revealed for the rest of the scene so it can be read.
  const scale  = useTransform(progress, [0.78, 0.86], [0.82, 1.0]);
  const radius = useTransform(progress, [0.78, 0.86], [40, 16]);
  const blur   = useTransform(progress, [0.78, 0.84], [10, 0]);
  // 3D entry tilt; frame leans forward as it lands.
  const rotateX = useTransform(progress, [0.78, 0.86], [12, 0]);
  const rotateY = useTransform(progress, [0.78, 0.86], [-6, 0]);

  return (
    <motion.div
      style={{
        scale, borderRadius: radius,
        rotateX, rotateY,
        transformPerspective: 1400,
        filter: useTransform(blur, (b) => `blur(${b}px)`) as any,
      }}
      className="relative w-[88vw] max-w-5xl aspect-[16/10] overflow-hidden border border-white/15 shadow-[0_60px_140px_-20px_rgba(99,102,241,0.5)]"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-ink-900 to-ink-950" />
      <div className="absolute -top-16 -right-16 h-72 w-72 rounded-full bg-brand-500/30 blur-3xl pointer-events-none" />
      <div className="absolute -bottom-16 -left-16 h-72 w-72 rounded-full bg-pink-500/30 blur-3xl pointer-events-none" />

      <div className="relative z-10 flex items-center gap-2 px-4 py-2.5 border-b border-white/10 bg-white/[0.04]">
        <span className="h-2.5 w-2.5 rounded-full bg-rose-400/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-amber-400/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
        <span className="ml-3 text-[11px] text-slate-400 tracking-wide">creative.ai · Predict</span>
        <span className="ml-auto text-[10px] text-slate-500">creative #500003</span>
      </div>

      <div className="relative z-10 flex h-[calc(100%-44px)]">
        {/* LEFT: real creative image */}
        <div className="flex-[5] p-4 flex flex-col gap-3 min-w-0">
          <div className="text-[10px] text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <ImageIcon className="h-3 w-3" /> creative
          </div>
          <div className="relative flex-1 rounded-lg overflow-hidden bg-black/40 border border-white/10">
            <img
              src="/assets/creative_500003.png"
              alt="ad creative"
              className="absolute inset-0 h-full w-full object-cover"
              onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
            />
            <div className="absolute inset-x-0 bottom-0 p-2.5 bg-gradient-to-t from-black/85 to-transparent">
              <div className="text-[10px] text-white/80 capitalize">gaming · rewarded video · 15s</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <span className="text-[10px] px-2 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300 text-center">
              theme · adventure
            </span>
            <span className="text-[10px] px-2 py-1 rounded-md bg-white/5 border border-white/10 text-slate-300 text-center">
              hook · gameplay
            </span>
          </div>
        </div>

        {/* RIGHT: predictions */}
        <div className="flex-[6] p-4 flex flex-col gap-3 min-w-0 border-l border-white/5">
          {/* Health */}
          <div className="rounded-xl bg-white/[0.04] border border-white/10 p-4 relative overflow-hidden">
            <div className="absolute -top-10 -right-6 h-28 w-28 rounded-full bg-brand-500/30 blur-2xl pointer-events-none" />
            <div className="relative flex items-end justify-between gap-3">
              <div>
                <div className="text-[10px] text-slate-400 uppercase tracking-wider">creative health score</div>
                <div className="mt-1 flex items-end gap-1">
                  <CountingNumber to={87} className="text-4xl font-extrabold tabular-nums leading-none" />
                  <span className="text-slate-500 text-sm mb-0.5">/100</span>
                </div>
              </div>
              <div className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-300 bg-emerald-500/15 rounded-full px-2.5 py-1 border border-emerald-500/30">
                <Target className="h-2.5 w-2.5" /> SCALE
              </div>
            </div>
            <div className="mt-3 h-1.5 rounded-full bg-white/10 overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-brand-500 to-pink-500"
                initial={{ width: 0 }} whileInView={{ width: "87%" }}
                viewport={{ once: true }} transition={{ duration: 1.4, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Predicted status */}
          <div className="rounded-xl bg-white/[0.04] border border-white/10 p-4 flex-1 min-h-0">
            <div className="flex items-center justify-between">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider">predicted status</div>
              <div className="text-[10px] text-emerald-300 font-bold uppercase tracking-wider">top performer</div>
            </div>
            <div className="mt-3 space-y-2">
              {[
                { label: "top performer",  v: 0.71, color: "bg-emerald-500/70", text: "text-emerald-300" },
                { label: "stable",         v: 0.21, color: "bg-sky-500/70",     text: "text-sky-300" },
                { label: "fatigued",       v: 0.06, color: "bg-amber-500/70",   text: "text-amber-300" },
                { label: "underperformer", v: 0.02, color: "bg-rose-500/70",    text: "text-rose-300" },
              ].map((row, i) => (
                <div key={row.label} className="text-[10px]">
                  <div className="flex justify-between mb-0.5">
                    <span className={`${row.text} capitalize`}>{row.label}</span>
                    <span className="tabular-nums text-slate-400">{(row.v * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1 rounded-full bg-white/10">
                    <motion.div
                      className={`h-full rounded-full ${row.color}`}
                      initial={{ width: 0 }} whileInView={{ width: `${row.v * 100}%` }}
                      viewport={{ once: true }} transition={{ duration: 0.8, delay: 0.3 + i * 0.08 }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI tags */}
          <div className="flex flex-wrap gap-1.5">
            {[
              { l: "✓ Strong CTA contrast",   c: "emerald" },
              { l: "✓ Brand visible <2s",     c: "emerald" },
              { l: "✗ Text density too high", c: "rose" },
              { l: "✦ Try warm palette",      c: "amber" },
            ].map((t, i) => (
              <motion.span
                key={t.l}
                initial={{ opacity: 0, y: 6 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: 0.4 + i * 0.08 }}
                className={`text-[10px] px-2 py-1 rounded-full border ${tagColor(t.c)}`}
              >
                {t.l}
              </motion.span>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function tagColor(c: string) {
  switch (c) {
    case "emerald": return "bg-emerald-500/10 text-emerald-300 border-emerald-500/30";
    case "rose":    return "bg-rose-500/10 text-rose-300 border-rose-500/30";
    case "amber":   return "bg-amber-500/10 text-amber-300 border-amber-500/30";
    case "brand":   return "bg-brand-500/15 text-brand-200 border-brand-500/30";
    default:        return "bg-white/5 text-slate-300 border-white/10";
  }
}

function ProgressDot({ progress, threshold }: { progress: MotionValue<number>; threshold: number }) {
  const opacity = useTransform(progress, [threshold, threshold + 0.05], [0.25, 1]);
  const scale = useTransform(progress, [threshold, threshold + 0.05], [1, 1.5]);
  return (
    <motion.span style={{ opacity, scale }} className="h-1.5 w-1.5 rounded-full bg-white" />
  );
}

function CountingNumber({ to, className = "" }: { to: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!inView) return;
    let raf = 0; const start = performance.now(); const dur = 1300;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      setVal(Math.round(to * (1 - Math.pow(1 - t, 3))));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, to]);
  return <span ref={ref} className={className}>{val}</span>;
}

// useTemplate: stitch a string with motion values reactively
function useTemplate(strings: string[], values: MotionValue<number>[]): MotionValue<string> {
  const out = useMotionValue("");
  useEffect(() => {
    const update = () => {
      let s = "";
      for (let i = 0; i < strings.length; i++) {
        s += strings[i];
        if (i < values.length) s += values[i].get();
      }
      out.set(s);
    };
    const unsubs = values.map((v) => v.on("change", update));
    update();
    return () => { unsubs.forEach((u) => u()); };
  }, [strings, values, out]);
  return out;
}

/* ============================================================
   3. FEATURES MARQUEE — vertical scroll → horizontal pin-row
   ============================================================ */
function FeaturesMarquee() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress: rawP } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const scrollYProgress = useSpring(rawP, { stiffness: 90, damping: 22, mass: 0.4 });
  const x = useTransform(scrollYProgress, [0, 1], ["0%", "-66%"]);
  // Subtle vertical drift on the cards as they pan — adds a sense of depth
  const cardY = useTransform(scrollYProgress, [0, 0.5, 1], [10, -10, 10]);

  const features: any[] = [
    {
      icon: ScanSearch, title: "Design errors detected", visual: "scan",
      body: "Every visual lever (CTA contrast, brand visibility, focal hierarchy) flagged automatically.",
      stat: "94", suffix: "%", statLabel: "AUC top performer",
      tint: "from-brand-500/30 to-transparent",
    },
    {
      icon: BarChart3, title: "Marketing diagnostics", visual: "bench",
      body: "Match your creative against the 1,076 creative benchmark in its vertical.",
      stat: "0.677", suffix: "", statLabel: "test macro-F1",
      tint: "from-pink-500/30 to-transparent",
    },
    {
      icon: Wand2, title: "One click AI rewrite", visual: "wand",
      body: "A redesigned variant that fixes every flagged weakness. Keep it or iterate.",
      stat: "1", suffix: "-click", statLabel: "to a better ad",
      tint: "from-emerald-500/30 to-transparent",
    },
    {
      icon: PencilLine, title: "Live drawing coach", visual: "draw",
      body: "Sketch over your creative. The AI streams live recommendations as you draw.",
      stat: "<1", suffix: "s", statLabel: "live tip latency",
      tint: "from-cyan-500/30 to-transparent",
    },
  ];

  return (
    <section ref={ref} className="relative min-h-[180vh]">
      <div className="sticky top-0 h-screen flex flex-col justify-center overflow-hidden">
        <motion.div
          initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }} transition={{ duration: 0.6 }}
          className="text-center px-6"
        >
          <span className="stat-pill"><Sparkles className="h-3 w-3 text-brand-300" /> what you get</span>
          <h2 className="mt-4 font-display text-3xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-[0.95]">
            One co pilot for every <span className="bg-gradient-to-r from-brand-300 to-pink-300 bg-clip-text text-transparent">creative decision</span>
          </h2>
          <p className="mt-3 text-sm text-slate-400">scroll to slide ↓</p>
        </motion.div>

        {/* horizontal lane */}
        <motion.div style={{ x }} className="mt-10 flex gap-6 px-[12vw] will-change-transform">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              style={{ y: cardY, rotate: i % 2 ? 0.6 : -0.6 }}
              whileHover={{ y: -8, scale: 1.02 }}
              transition={{ type: "spring", stiffness: 200, damping: 22 }}
              className="card-surface p-7 w-[80vw] sm:w-[60vw] md:w-[40vw] lg:w-[30vw] shrink-0 relative overflow-hidden"
            >
              <div className={`absolute -top-32 -right-20 h-72 w-72 rounded-full bg-gradient-to-br ${f.tint} blur-3xl pointer-events-none`} />
              <div className="relative">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/10">
                  <f.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-4 text-xl font-bold">{f.title}</h3>
                <p className="mt-2 text-sm text-slate-300 leading-relaxed">{f.body}</p>
                <div className="mt-5 flex items-baseline gap-2">
                  <span className="text-4xl font-extrabold tabular-nums tracking-tight">{f.stat}</span>
                  <span className="text-2xl font-bold text-white">{f.suffix}</span>
                  <span className="text-xs text-slate-400 ml-2 uppercase tracking-wider">{f.statLabel}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

/* ============================================================
   4. HOW IT WORKS
   ============================================================ */
function HowItWorks() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.25 });

  const steps = [
    {
      title: "Drop your ad",
      body: "Upload a screenshot of any campaign creative. PNG, JPG, or WebP.",
      tag: "<300ms",
      tagLabel: "to upload",
      illustration: <DropIllustration />,
      tint: "from-brand-500/30 to-cyan-500/10",
      ring: "ring-brand-500/40",
    },
    {
      title: "AI analyzes it",
      body: "We extract visual metadata, score it with the trained ensemble, and an LLM finds every weak spot.",
      tag: "6-model",
      tagLabel: "ensemble",
      illustration: <ScanIllustration />,
      tint: "from-pink-500/30 to-rose-500/10",
      ring: "ring-pink-500/40",
    },
    {
      title: "Get a fix",
      body: "One click for an AI rewritten variant, or draw your own with live AI coaching as you sketch.",
      tag: "1-click",
      tagLabel: "to a better ad",
      illustration: <FixIllustration />,
      tint: "from-emerald-500/30 to-amber-500/10",
      ring: "ring-emerald-500/40",
    },
  ];

  return (
    <section id="how-it-works" ref={ref} className="container-narrow py-24 sm:py-32 relative">
      <motion.div
        initial={{ opacity: 0, y: 16 }} animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6 }} className="text-center"
      >
        <span className="stat-pill"><Target className="h-3 w-3 text-brand-300" /> how it works</span>
        <h2 className="mt-4 font-display text-3xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-[0.95]">
          Three steps to a <span className="bg-gradient-to-r from-brand-300 via-pink-300 to-emerald-300 bg-clip-text text-transparent">better ad</span>
        </h2>
        <p className="mt-4 text-slate-400 text-sm sm:text-base max-w-xl mx-auto">
          From upload to AI-rewrite in under 30 seconds. No setup, no model training, no signup.
        </p>
      </motion.div>

      <div className="mt-16 relative">
        {/* connecting flow — full path with animated draw + 3 nodes */}
        <svg
          className="hidden md:block pointer-events-none absolute left-0 right-0 top-[5.5rem] mx-auto"
          width="100%" height="80" viewBox="0 0 1000 80" preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="flowGrad" x1="0" y1="0" x2="1000" y2="0" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="50%" stopColor="#ec4899" />
              <stop offset="100%" stopColor="#34d399" />
            </linearGradient>
          </defs>
          <motion.path
            d="M 60 40 Q 200 10, 333 40 T 666 40 T 940 40"
            fill="none" stroke="url(#flowGrad)" strokeWidth="2"
            strokeDasharray="3 5"
            initial={{ pathLength: 0 }}
            animate={inView ? { pathLength: 1 } : {}}
            transition={{ duration: 1.8, ease: "easeInOut", delay: 0.3 }}
          />
        </svg>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {steps.map((s, i) => (
            <StepCard key={s.title} step={s} idx={i} inView={inView} />
          ))}
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0 }} animate={inView ? { opacity: 1 } : {}}
        transition={{ duration: 0.6, delay: 1.4 }}
        className="mt-20 flex flex-col items-center gap-3"
      >
        <Link to="/predict" className="btn-primary group text-base">
          <Zap className="h-4 w-4" />
          Try it now
          <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
        </Link>
        <span className="uppercase tracking-[0.3em] text-[10px] text-slate-500">no signup · runs in your browser</span>
      </motion.div>
    </section>
  );
}

function StepCard({ step, idx, inView }: { step: any; idx: number; inView: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay: idx * 0.15 + 0.2, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -8 }}
      className="card-surface relative overflow-hidden p-6 group"
    >
      <div className={`absolute -top-32 -right-20 h-72 w-72 rounded-full bg-gradient-to-br ${step.tint} blur-3xl pointer-events-none`} />

      {/* Step number — big floating badge */}
      <div className="relative flex items-start justify-between">
        <motion.div
          initial={{ scale: 0, rotate: -45 }}
          animate={inView ? { scale: 1, rotate: 0 } : {}}
          transition={{ delay: idx * 0.15 + 0.4, type: "spring", stiffness: 200, damping: 15 }}
          className={`relative h-12 w-12 rounded-2xl bg-white/10 ring-1 ${step.ring} grid place-items-center backdrop-blur shadow-lg`}
        >
          <span className="font-display text-2xl font-extrabold tabular-nums leading-none">{idx + 1}</span>
        </motion.div>
        <div className="text-right">
          <div className="font-display text-xl font-extrabold tracking-tight tabular-nums">{step.tag}</div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400">{step.tagLabel}</div>
        </div>
      </div>

      {/* Mini illustration */}
      <div className="relative mt-6 rounded-xl bg-black/30 border border-white/10 aspect-[4/3] overflow-hidden">
        {step.illustration}
      </div>

      {/* Title + body */}
      <div className="relative mt-5">
        <h3 className="font-display text-2xl font-extrabold tracking-tight">{step.title}</h3>
        <p className="mt-2 text-sm text-slate-300 leading-relaxed">{step.body}</p>
      </div>
    </motion.div>
  );
}

/* --- Mini illustrations for each step --- */
function DropIllustration() {
  return (
    <div className="absolute inset-0 grid place-items-center">
      <motion.div
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut" }}
        className="relative w-24 h-32 rounded-lg overflow-hidden border border-white/15 shadow-lg"
        style={{ background: "linear-gradient(135deg, rgba(99,102,241,0.5), rgba(236,72,153,0.5))" }}
      >
        <div className="absolute inset-0 grid place-items-center text-white/90">
          <ImageIcon className="h-6 w-6" />
        </div>
        <div className="absolute inset-x-2 bottom-2 h-1.5 rounded-full bg-white/30 overflow-hidden">
          <motion.div
            className="h-full bg-white"
            initial={{ width: "0%" }}
            animate={{ width: ["0%", "100%", "100%"] }}
            transition={{ duration: 2.6, repeat: Infinity, times: [0, 0.6, 1] }}
          />
        </div>
      </motion.div>
      {/* Dashed dropzone outline */}
      <div className="absolute inset-4 rounded-xl border-2 border-dashed border-white/15 pointer-events-none" />
      {/* Floating up-arrow cue */}
      <motion.div
        animate={{ y: [-2, -10, -2], opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        className="absolute bottom-4 right-4 text-brand-300"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 19V5M5 12l7-7 7 7" />
        </svg>
      </motion.div>
    </div>
  );
}

function ScanIllustration() {
  const boxes = [
    { x: "18%", y: "22%", w: "32%", h: "20%", label: "CTA", delay: 0.0, color: "rgba(244,63,94,0.9)" },
    { x: "55%", y: "30%", w: "30%", h: "30%", label: "logo", delay: 0.5, color: "rgba(52,211,153,0.9)" },
    { x: "20%", y: "55%", w: "60%", h: "15%", label: "text density", delay: 1.0, color: "rgba(251,191,36,0.9)" },
  ];
  return (
    <div className="absolute inset-0">
      {/* Mock creative background */}
      <div className="absolute inset-0" style={{ background: "linear-gradient(135deg, rgba(236,72,153,0.35), rgba(99,102,241,0.35), rgba(34,211,238,0.2))" }}>
        <div className="absolute inset-x-4 top-5 h-3 rounded bg-white/30" />
        <div className="absolute inset-x-4 top-10 h-2 rounded bg-white/15 w-1/2" />
        <div className="absolute left-4 right-4 bottom-5 h-5 rounded-full bg-white/40" />
      </div>

      {/* Detection boxes */}
      {boxes.map((b, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: [0, 1, 1, 0], scale: [0.85, 1, 1, 0.85] }}
          transition={{ duration: 3.4, repeat: Infinity, delay: b.delay, times: [0, 0.15, 0.85, 1] }}
          className="absolute rounded-md"
          style={{ left: b.x, top: b.y, width: b.w, height: b.h, border: `1.5px solid ${b.color}`, boxShadow: `0 0 12px ${b.color}` }}
        >
          <span className="absolute -top-4 left-0 text-[9px] font-bold uppercase tracking-wider text-white/90 bg-black/60 backdrop-blur px-1 rounded">
            {b.label}
          </span>
        </motion.div>
      ))}

      {/* Scanning line */}
      <motion.div
        className="absolute inset-x-0 h-12 pointer-events-none"
        style={{ background: "linear-gradient(to bottom, transparent, rgba(99,102,241,0.5), transparent)" }}
        animate={{ y: ["-30%", "120%"] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}

function FixIllustration() {
  return (
    <div className="absolute inset-0 flex">
      {/* BEFORE half */}
      <div className="relative flex-1 overflow-hidden" style={{ background: "linear-gradient(135deg, rgba(244,63,94,0.35), rgba(244,63,94,0.15))" }}>
        <div className="absolute inset-x-2 top-4 h-2.5 rounded bg-white/20 w-1/2" />
        <div className="absolute inset-x-2 top-9 h-1.5 rounded bg-white/15 w-2/3" />
        <div className="absolute left-2 right-2 bottom-3 h-3.5 rounded bg-white/25" />
        <div className="absolute top-2 left-2 text-[8px] font-bold uppercase tracking-wider text-rose-200 bg-rose-500/30 backdrop-blur px-1 rounded">before</div>
        <div className="absolute bottom-2 right-2 text-[10px] font-bold tabular-nums text-rose-200">38</div>
      </div>
      {/* Wand divider */}
      <motion.div
        animate={{ y: [0, -4, 0] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10 h-9 w-9 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-400 grid place-items-center shadow-[0_0_20px_rgba(52,211,153,0.6)]"
      >
        <Wand2 className="h-4 w-4 text-ink-950" />
      </motion.div>
      {/* AFTER half */}
      <div className="relative flex-1 overflow-hidden" style={{ background: "linear-gradient(135deg, rgba(52,211,153,0.35), rgba(34,211,238,0.2))" }}>
        <div className="absolute inset-x-2 top-3 h-3 rounded bg-white/40 w-3/4" />
        <div className="absolute inset-x-2 top-8 h-2 rounded bg-white/30 w-1/2" />
        <div className="absolute left-2 right-2 bottom-2 h-4 rounded bg-emerald-300/80" />
        <div className="absolute top-2 right-2 text-[8px] font-bold uppercase tracking-wider text-emerald-200 bg-emerald-500/30 backdrop-blur px-1 rounded">after</div>
        <div className="absolute bottom-2 left-2 text-[10px] font-bold tabular-nums text-emerald-200">87</div>
      </div>
      {/* Floating sparkles */}
      {[
        { x: "20%", y: "25%", d: 0 },
        { x: "78%", y: "40%", d: 0.5 },
        { x: "65%", y: "75%", d: 1 },
      ].map((s, i) => (
        <motion.span
          key={i}
          className="absolute text-amber-200"
          style={{ left: s.x, top: s.y }}
          animate={{ scale: [0, 1, 0], rotate: [0, 90, 180] }}
          transition={{ duration: 2, repeat: Infinity, delay: s.d, ease: "easeInOut" }}
        >
          ✦
        </motion.span>
      ))}
    </div>
  );
}

/* ============================================================
   5. FINAL CTA
   ============================================================ */
function FinalCTA() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });
  return (
    <section ref={ref} className="container-narrow pb-28">
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.96 }}
        animate={inView ? { opacity: 1, y: 0, scale: 1 } : {}}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="card-surface relative overflow-hidden p-10 sm:p-16 text-center"
      >
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-brand-500/15 via-transparent to-pink-500/15" />
        <motion.div
          className="pointer-events-none absolute -top-24 -right-16 h-72 w-72 rounded-full bg-brand-500/30 blur-3xl"
          animate={{ scale: [1, 1.15, 1] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="pointer-events-none absolute -bottom-24 -left-16 h-72 w-72 rounded-full bg-pink-500/30 blur-3xl"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />
        <div className="relative">
          <Sparkles className="mx-auto h-9 w-9 text-brand-300" />
          <h2 className="mt-3 font-display text-3xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-[0.95]">
            Stop guessing.<br />
            <span className="bg-gradient-to-r from-brand-300 to-pink-300 bg-clip-text text-transparent">
              Start scoring.
            </span>
          </h2>
          <p className="mt-4 max-w-xl mx-auto text-sm sm:text-base text-slate-300">
            Drop one creative. See the score, the failure cases, and the AI fix in under 30 seconds.
          </p>
          <motion.div
            animate={{ scale: [1, 1.04, 1] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
            className="mt-8 inline-block"
          >
            <Link to="/predict" className="btn-primary text-base px-8 py-3.5 group">
              <Zap className="h-5 w-5" />
              Start now, it's free
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
          </motion.div>
          <p className="mt-4 text-[11px] uppercase tracking-[0.25em] text-slate-500">
            No signup · Runs in your browser
          </p>
        </div>
      </motion.div>
    </section>
  );
}
