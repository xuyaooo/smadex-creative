import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight, Sparkles, LineChart, Activity, Brain, ShieldCheck,
  Zap, Target,
} from "lucide-react";
import Logo from "../components/Logo";

const features = [
  {
    icon: LineChart,
    title: "Status Classifier",
    body: "Soft-vote ensemble of 5 SOTA models — XGBoost (5-seed bag), LightGBM, CatBoost, HistGBM, LogReg. Test macro-F1 0.677 with AUC 0.94 on top performers.",
    accent: "from-emerald-500/30 to-transparent",
  },
  {
    icon: Activity,
    title: "Health Score 0-100",
    body: "Decomposed score combining calibrated probabilities + percentile-within-vertical CTR & ROAS + anti-fatigue. Maps to Scale / Maintain / Watch / Pause actions.",
    accent: "from-brand-500/30 to-transparent",
  },
  {
    icon: Brain,
    title: "Fatigue Forecast",
    body: "4-bucket LightGBM classifier (Never / Late / Standard / Early). Plus a lifecycle archetype model that predicts CTR-decay shape from launch features.",
    accent: "from-pink-500/30 to-transparent",
  },
  {
    icon: ShieldCheck,
    title: "Leakage-Free Splits",
    body: "13 future-data columns dropped, group-aware StratifiedGroupKFold by campaign_id, zero campaign overlap across train/val/test (717/143/216).",
    accent: "from-amber-500/30 to-transparent",
  },
];

const stats = [
  { label: "Test macro-F1", value: "0.677" },
  { label: "AUC top_performer", value: "0.94" },
  { label: "AUC underperformer", value: "0.98" },
  { label: "Health Score Spearman", value: "0.45" },
  { label: "Train n", value: "860" },
  { label: "Test n", value: "216" },
];

export default function HomePage() {
  return (
    <main className="pt-16">
      {/* HERO */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 -z-10">
          <div className="absolute -top-24 -left-20 h-[28rem] w-[28rem] rounded-full bg-brand-500/30 blur-3xl animate-blob" />
          <div className="absolute top-40 -right-24 h-[26rem] w-[26rem] rounded-full bg-pink-500/25 blur-3xl animate-blob [animation-delay:-6s]" />
          <div className="absolute bottom-0 left-1/3 h-[22rem] w-[22rem] rounded-full bg-cyan-400/15 blur-3xl animate-blob [animation-delay:-12s]" />
        </div>

        <div className="container-narrow section-pad flex flex-col items-center text-center">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-4 py-1.5 text-xs font-medium text-slate-200 backdrop-blur"
          >
            <Sparkles className="h-3.5 w-3.5 text-brand-300" />
            Smadex Creative Intelligence Challenge — PoC
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="mt-10"
          >
            <Logo size={72} className="drop-shadow-[0_10px_30px_rgba(99,102,241,0.45)]" />
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 }}
            className="mt-6 bg-gradient-to-br from-white via-white to-white/60 bg-clip-text text-5xl font-extrabold tracking-tight text-transparent sm:text-6xl md:text-7xl"
          >
            Creative Intelligence
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.25 }}
            className="mt-5 max-w-xl text-balance text-base text-slate-300 sm:text-lg"
          >
            Turn raw ad creatives into <span className="text-white font-semibold">Scale</span>,{" "}
            <span className="text-white font-semibold">Pause</span> or{" "}
            <span className="text-white font-semibold">Pivot</span> recommendations — backed by a
            6-model ensemble, fatigue forecast, and a calibrated 0–100 Health Score.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.35 }}
            className="mt-10 flex flex-col sm:flex-row items-center gap-3"
          >
            <Link to="/stats" className="btn-primary group">
              See the numbers
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link to="/predict" className="btn-ghost">
              <Zap className="h-4 w-4" /> Predict a creative
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="mt-14 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 w-full max-w-4xl"
          >
            {stats.map((s) => (
              <div key={s.label} className="card-surface px-4 py-3 text-left">
                <div className="text-xl font-bold tracking-tight">{s.value}</div>
                <div className="text-[11px] uppercase tracking-wider text-slate-400 mt-0.5">{s.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" className="container-narrow py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.05 }}
              className="card-surface relative overflow-hidden p-7"
            >
              <div className={`absolute -top-32 -right-20 h-72 w-72 rounded-full bg-gradient-to-br ${f.accent} blur-3xl pointer-events-none`} />
              <div className="relative">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/10">
                  <f.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-4 text-xl font-bold">{f.title}</h3>
                <p className="mt-2 text-sm text-slate-300 leading-relaxed">{f.body}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CALL TO ACTION */}
      <section className="container-narrow py-16">
        <div className="card-surface relative overflow-hidden p-10 text-center">
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-brand-500/10 via-transparent to-pink-500/10" />
          <div className="relative">
            <Target className="mx-auto h-9 w-9 text-brand-300" />
            <h2 className="mt-3 text-2xl sm:text-3xl font-bold">Built honestly, evaluated rigorously</h2>
            <p className="mt-3 max-w-2xl mx-auto text-sm text-slate-300">
              Bootstrap 95% CIs on every per-class F1. Group-aware splits. No temperature scaling
              (raw probabilities). Honest caveats documented. Reproducible end-to-end in 4 minutes.
            </p>
            <div className="mt-6 flex flex-col sm:flex-row justify-center gap-3">
              <Link to="/stats" className="btn-primary">Open the dashboard</Link>
              <Link to="/explorer" className="btn-ghost">Explore predictions</Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
