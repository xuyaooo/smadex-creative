import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import {
  loadMetrics, loadEvalReport, loadPredictions,
  type FinalMetrics, type EvalReport, type Prediction, STATUS_COLORS, VERTICAL_COLORS,
} from "../lib/data";
import {
  Trophy, Activity, BadgeCheck, Layers, Database, Filter, Brain,
  Workflow, ScanSearch, AlertTriangle, CheckCircle2, Sparkles,
  Code, FlaskConical, Microscope, BookOpen, ArrowDown, Zap, Wand2,
} from "lucide-react";

const CLASS_NAMES = ["fatigued", "stable", "top_performer", "underperformer"] as const;

/* ============================================================
   Page
   ============================================================ */
export default function StatsPage() {
  const [metrics, setMetrics] = useState<FinalMetrics | null>(null);
  const [evalReport, setEval] = useState<EvalReport | null>(null);
  const [preds, setPreds] = useState<Prediction[] | null>(null);

  useEffect(() => {
    Promise.all([loadMetrics(), loadEvalReport(), loadPredictions()]).then(
      ([m, e, p]) => { setMetrics(m); setEval(e); setPreds(p); },
    );
  }, []);

  if (!metrics || !evalReport || !preds) {
    return <main className="pt-28 container-narrow"><p className="text-slate-400">loading…</p></main>;
  }

  const test = preds.filter((p) => p.split === "test");
  const counts = test.reduce<Record<string, number>>((acc, p) => {
    acc[p.true_status] = (acc[p.true_status] || 0) + 1; return acc;
  }, {});

  return (
    <main className="pt-24 pb-24 relative overflow-hidden">
      {/* atmospheric backdrop */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-20 h-[28rem] w-[28rem] rounded-full bg-brand-500/15 blur-3xl animate-blob" />
        <div className="absolute top-40 -right-24 h-[22rem] w-[22rem] rounded-full bg-pink-500/10 blur-3xl animate-blob [animation-delay:-6s]" />
      </div>

      <div className="container-narrow space-y-24">
        <Intro />
        <Chapter1Problem />
        <Chapter2Dataset preds={preds} />
        <Chapter3Splits />
        <Chapter4Features />
        <Chapter5VLM />
        <Chapter7Results metrics={metrics} evalReport={evalReport} />
        <Chapter8Findings counts={counts} preds={preds} testN={test.length} />
        <Chapter9Caveats />
      </div>
    </main>
  );
}

/* ============================================================
   Intro
   ============================================================ */
function Intro() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
      className="text-center pt-4"
    >
      <span className="stat-pill"><BookOpen className="h-3 w-3 text-brand-300" /> the story</span>
      <h1 className="mt-4 font-display text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight leading-[0.95] bg-gradient-to-br from-white via-white to-white/60 bg-clip-text text-transparent">
        How we built it
      </h1>
      <p className="mt-5 text-slate-300 text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
        From 1,076 raw ad creatives to a calibrated, leakage-free, six-model ensemble
        that predicts whether to scale, maintain, watch, or pause a campaign.
      </p>
      <motion.div
        animate={{ y: [0, 8, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        className="mt-8 inline-flex flex-col items-center text-slate-500 text-xs uppercase tracking-[0.3em]"
      >
        <ArrowDown className="h-5 w-5 mb-2" /> read the full story
      </motion.div>
    </motion.section>
  );
}

/* ============================================================
   Chapter wrapper — animates reveal + handles the chapter number/icon
   ============================================================ */
function Chapter({
  number, icon: Icon, label, title, subtitle, children, accent = "brand",
}: {
  number: string;
  icon: any;
  label: string;
  title: React.ReactNode;
  subtitle: string;
  children: React.ReactNode;
  accent?: "brand" | "pink" | "emerald" | "amber" | "cyan" | "rose";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.2 });
  const accentMap = {
    brand:   "text-brand-300 bg-brand-500/10 border-brand-500/30",
    pink:    "text-pink-300 bg-pink-500/10 border-pink-500/30",
    emerald: "text-emerald-300 bg-emerald-500/10 border-emerald-500/30",
    amber:   "text-amber-300 bg-amber-500/10 border-amber-500/30",
    cyan:    "text-cyan-300 bg-cyan-500/10 border-cyan-500/30",
    rose:    "text-rose-300 bg-rose-500/10 border-rose-500/30",
  };
  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
      className="scroll-mt-28"
    >
      <div className="flex items-start gap-4">
        <div className={`shrink-0 inline-flex items-center justify-center h-10 w-10 sm:h-12 sm:w-12 rounded-2xl border ${accentMap[accent]} font-display font-extrabold tabular-nums`}>
          {number}
        </div>
        <div className="flex-1 min-w-0">
          <div className={`inline-flex items-center gap-1.5 text-[11px] uppercase tracking-[0.2em] ${accentMap[accent].split(" ")[0]}`}>
            <Icon className="h-3 w-3" /> {label}
          </div>
          <h2 className="mt-2 font-display text-2xl sm:text-4xl font-extrabold tracking-tight leading-[1.05]">{title}</h2>
          <p className="mt-2 text-slate-300 text-sm sm:text-base leading-relaxed max-w-2xl">{subtitle}</p>
        </div>
      </div>
      <div className="mt-6 sm:ml-16">{children}</div>
    </motion.section>
  );
}

/* ============================================================
   Chapter 1 — The problem
   ============================================================ */
function Chapter1Problem() {
  return (
    <Chapter
      number="01" icon={AlertTriangle} accent="rose"
      label="the problem"
      title={<>Most ads die quietly.<br /><span className="text-rose-300">By the time the dashboard turns red, you've already burnt half the budget.</span></>}
      subtitle="Performance marketers can't tell which creatives will scale, fatigue, or stagnate until weeks of spend have already left the door. We wanted a model that could read the early signal."
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <ProblemCard icon={AlertTriangle} title="Fatigue creeps in" body="CTR collapses gradually. By day 14 a 'top performer' may be a budget sink." />
        <ProblemCard icon={Zap} title="Wasted spend" body="A single fatigued creative can burn $18k+ before triggering a manual review." />
        <ProblemCard icon={ScanSearch} title="No design feedback" body="Marketers see numbers; designers see images. The link between them is invisible." />
      </div>
    </Chapter>
  );
}

function ProblemCard({ icon: Icon, title, body }: { icon: any; title: string; body: string }) {
  return (
    <div className="card-surface p-5">
      <div className="inline-flex items-center justify-center h-9 w-9 rounded-lg bg-rose-500/10 text-rose-300 border border-rose-500/30 mb-3">
        <Icon className="h-4 w-4" />
      </div>
      <h3 className="font-display font-bold text-base">{title}</h3>
      <p className="mt-1.5 text-sm text-slate-300 leading-relaxed">{body}</p>
    </div>
  );
}

/* ============================================================
   Chapter 2 — The dataset
   ============================================================ */
function Chapter2Dataset({ preds }: { preds: Prediction[] }) {
  // Build vertical distribution from full predictions corpus
  const verticalCounts = preds.reduce<Record<string, number>>((acc, p) => {
    acc[p.vertical] = (acc[p.vertical] || 0) + 1; return acc;
  }, {});
  const verticals = Object.entries(verticalCounts).sort(([, a], [, b]) => b - a);
  const maxV = Math.max(...verticals.map(([, n]) => n));

  return (
    <Chapter
      number="02" icon={Database} accent="brand"
      label="the dataset"
      title={<>1,076 creatives. <span className="text-brand-300">36 advertisers.</span> 6 verticals.</>}
      subtitle="The Smadex Creative Intelligence Challenge dataset: synthetic mobile ad creatives with full lifecycle telemetry and four ground-truth status classes."
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <DatasetStat n="1,076" label="creatives" />
        <DatasetStat n="192k" label="daily fact rows" />
        <DatasetStat n="36"   label="advertisers" />
        <DatasetStat n="4"    label="status classes" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Vertical distribution */}
        <div className="card-surface p-5">
          <h3 className="font-display font-bold text-sm mb-1">Creatives per vertical</h3>
          <p className="text-xs text-slate-400 mb-4">Roughly balanced across the six business verticals.</p>
          <div className="space-y-2">
            {verticals.map(([v, n]) => (
              <div key={v}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="capitalize font-medium" style={{ color: VERTICAL_COLORS[v] || "#cbd5e1" }}>
                    {v.replace("_", " ")}
                  </span>
                  <span className="text-slate-400 tabular-nums">{n}</span>
                </div>
                <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: VERTICAL_COLORS[v] || "#94a3b8", opacity: 0.75 }}
                    initial={{ width: 0 }} whileInView={{ width: `${(n / maxV) * 100}%` }}
                    viewport={{ once: true }} transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* What's in each row */}
        <div className="card-surface p-5">
          <h3 className="font-display font-bold text-sm mb-1">What's in each creative</h3>
          <p className="text-xs text-slate-400 mb-4">Rich tabular metadata + an image asset + 14-day daily lifecycle stats.</p>
          <div className="space-y-2.5 text-xs">
            <DataRow label="Creative attributes" value="vertical, format, theme, hook_type, dominant_color, emotional_tone, cta_text" />
            <DataRow label="Visual scores"        value="text_density, brand_visibility, clutter, novelty, motion (rubric, 0–1)" />
            <DataRow label="Counts & flags"       value="faces_count, product_count, has_price, has_discount_badge, has_gameplay, has_ugc_style" />
            <DataRow label="Image asset"          value="creative_<id>.png, 512×512" />
            <DataRow label="Daily telemetry"      value="impressions, clicks, spend, revenue, CTR, ROAS — by country × OS × day" />
            <DataRow label="Ground-truth status"  value="top_performer / stable / fatigued / underperformer" />
          </div>
        </div>
      </div>
    </Chapter>
  );
}

function DatasetStat({ n, label }: { n: string; label: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}
      className="card-surface p-4 text-center"
    >
      <div className="font-display text-3xl sm:text-4xl font-extrabold tabular-nums tracking-tight">{n}</div>
      <div className="mt-1 text-[10px] uppercase tracking-[0.2em] text-slate-400">{label}</div>
    </motion.div>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-3">
      <span className="text-slate-500 uppercase tracking-wider text-[10px] sm:w-32 shrink-0">{label}</span>
      <span className="text-slate-200">{value}</span>
    </div>
  );
}

/* ============================================================
   Chapter 3 — Splits / leakage
   ============================================================ */
function Chapter3Splits() {
  return (
    <Chapter
      number="03" icon={Filter} accent="amber"
      label="avoiding leakage"
      title={<>13 columns dropped.<br /><span className="text-amber-300">Zero campaign overlap across splits.</span></>}
      subtitle="The single biggest threat to creative-performance models is label leakage. Same creative, same campaign, two splits — and the model just memorizes."
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card-surface p-5">
          <h3 className="font-display font-bold text-sm mb-2">Future-data columns dropped</h3>
          <p className="text-xs text-slate-400 mb-3">These would have given the model the answer.</p>
          <div className="flex flex-wrap gap-1.5">
            {["overall_ctr", "overall_ipm", "overall_roas", "total_clicks", "total_spend", "total_revenue",
              "last_7d_ctr", "last_7d_ipm", "ctr_decay_pct", "perf_score", "creative_status",
              "lifecycle_archetype", "fatigue_label"].map((c) => (
                <span key={c} className="text-[10px] font-mono px-2 py-1 rounded bg-rose-500/10 border border-rose-500/30 text-rose-200 line-through">
                  {c}
                </span>
            ))}
          </div>
        </div>

        <div className="card-surface p-5">
          <h3 className="font-display font-bold text-sm mb-2">Group-aware splits</h3>
          <p className="text-xs text-slate-400 mb-4">StratifiedGroupKFold by <code className="px-1 rounded bg-white/5 font-mono">campaign_id</code>. No campaign appears in two splits.</p>
          <div className="space-y-2">
            <SplitRow label="train" n={717} pct={66.6} color="bg-brand-500/70" />
            <SplitRow label="val"   n={143} pct={13.3} color="bg-pink-500/70" />
            <SplitRow label="test"  n={216} pct={20.1} color="bg-emerald-500/70" />
          </div>
        </div>
      </div>

      <div className="mt-4 card-surface p-5">
        <h3 className="font-display font-bold text-sm mb-2">Why this matters</h3>
        <p className="text-sm text-slate-300 leading-relaxed">
          A predict-by-vertical-prior baseline gets ~0.30 macro-F1 for free, because verticals strongly
          predict status (gaming has more top performers than fintech). Our model adds <span className="text-emerald-300 font-semibold">+0.35 macro-F1</span> on top of that,
          mostly via early-life CTR aggregates from the first 7 days of telemetry. Without leakage-free splits, that lift would be illusory.
        </p>
      </div>
    </Chapter>
  );
}

function SplitRow({ label, n, pct, color }: { label: string; n: number; pct: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="font-medium text-slate-200">{label}</span>
        <span className="text-slate-400 tabular-nums">{n} · {pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          className={`h-full ${color}`}
          initial={{ width: 0 }} whileInView={{ width: `${pct}%` }}
          viewport={{ once: true }} transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
    </div>
  );
}

/* ============================================================
   Chapter 4 — Features
   ============================================================ */
function Chapter4Features() {
  const blocks = [
    { icon: Layers, title: "Tabular metadata", body: "OHE for vertical, format, color, tone, language, objective, kpi_goal, target_os, hq_region. Label-encoded for theme, hook, cta_text, age segment.", count: "9 OHE · 4 LE" },
    { icon: Activity, title: "Numeric scores",  body: "Visual rubric (text density, readability, brand visibility, clutter, novelty, motion) + counts (faces, products) + budget + duration.", count: "12 num" },
    { icon: BadgeCheck, title: "Binary flags",   body: "has_price, has_discount_badge, has_gameplay, has_ugc_style. Auto-extracted from the image when the user uploads via /predict.", count: "4 flags" },
    { icon: Workflow, title: "Early-life signal", body: "First-7-day aggregates over the daily fact table: imps, clicks, spend, revenue, CTR, IPM, ROAS, video completions, decay slope.", count: "9 num" },
    { icon: ScanSearch, title: "Visual rubric (LLM)", body: "OpenRouter Gemini scores 15 visual dimensions per image (hook clarity, CTA prominence, contrast, urgency, novelty, …) anchored 0–10.", count: "15 dim" },
    { icon: Brain, title: "CLIP image embeddings", body: "SigLIP-2 image encoder → 768d → PCA-64 reduction. Lets the model 'see' the creative even when metadata is sparse.", count: "768→64d" },
  ];
  return (
    <Chapter
      number="04" icon={Code} accent="cyan"
      label="feature engineering"
      title={<>Six feature families. <span className="text-cyan-300">All concatenated.</span></>}
      subtitle="The model sees structured metadata, early performance signal, an LLM-scored visual rubric, and a learned image embedding — fused into a single tabular row per creative."
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {blocks.map((b, i) => (
          <motion.div
            key={b.title}
            initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.5, delay: i * 0.05 }}
            className="card-surface p-5"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="inline-flex items-center justify-center h-9 w-9 rounded-lg bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                <b.icon className="h-4 w-4" />
              </div>
              <span className="font-mono text-[10px] text-slate-400 bg-white/5 border border-white/10 rounded px-1.5 py-0.5">{b.count}</span>
            </div>
            <h3 className="mt-3 font-display font-bold text-sm">{b.title}</h3>
            <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">{b.body}</p>
          </motion.div>
        ))}
      </div>
    </Chapter>
  );
}

/* ============================================================
   Chapter 5 — Personalized small VLM (teacher → student via SDFT)
   ============================================================ */
function Chapter5VLM() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.15 });

  return (
    <Chapter
      number="05" icon={Brain} accent="pink"
      label="three personalized models"
      title={<>We trained <span className="text-pink-300">three</span> personalized models for this task.</>}
      subtitle="A soft-voting tabular ensemble for prediction, a small VLM for analysis JSON, and a Flux edit fine-tune for the rebuild. Each does one thing well; the predict page chains all three."
    >
      {/* ─────────────  Visual divider above Model 1 ───────────── */}
      <div className="mb-3 flex items-center gap-3">
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
        <span className="text-[10px] uppercase tracking-[0.3em] text-slate-500 font-mono">1 of 3</span>
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
      </div>

      {/* ─────────────  MODEL 1: Soft-voting tabular ensemble  ───────────── */}
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-200 border border-emerald-500/30">
          MODEL 1
        </span>
        <h3 className="font-display text-lg font-extrabold tracking-tight">Soft-voting tabular ensemble · prediction</h3>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/5 text-slate-400 border border-white/10 ml-auto">
          5 boosters · macro-F1 0.677
        </span>
      </div>
      <p className="text-xs text-slate-400 mb-4 max-w-3xl">
        The prediction core. Five base learners trained on the same leakage-free splits, their calibrated probabilities averaged into one 4-class output (top performer / stable / fatigued / underperformer) plus a 0–100 health score.
      </p>

      <SoftVotingArchitecture />

      {/* ─────────────  Visual divider above Model 2 ───────────── */}
      <div className="my-8 flex items-center gap-3">
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
        <span className="text-[10px] uppercase tracking-[0.3em] text-slate-500 font-mono">2 of 3</span>
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
      </div>

      {/* ─────────────  MODEL 2: Personalized VLM  ───────────── */}
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-pink-500/15 text-pink-200 border border-pink-500/30">
          MODEL 2
        </span>
        <h3 className="font-display text-lg font-extrabold tracking-tight">Personalized VLM · analysis</h3>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/5 text-slate-400 border border-white/10 ml-auto">
          SmolVLM 2.2B · full FT + SDFT
        </span>
      </div>
      <p className="text-xs text-slate-400 mb-4 max-w-3xl">
        Teacher pseudo-labels from a Gemma-family model feed an SDFT loop that fully fine-tunes SmolVLM. At runtime we hit a fast Flash-tier endpoint for sub-second prompting on every creative — the JSON the predict page renders comes from this model.
      </p>

      {/* Pipeline diagram */}
      <div ref={ref} className="card-surface p-5 sm:p-6 relative overflow-hidden">
        <div className="absolute -top-24 -right-12 h-56 w-56 rounded-full bg-pink-500/15 blur-3xl pointer-events-none" />
        <div className="relative grid grid-cols-1 md:grid-cols-[1fr_auto_1fr_auto_1fr] gap-4 items-stretch">
          <PipeNode
            icon={Sparkles} title="Teacher" tag="Gemma family · Gemini 2.5"
            body="Large multimodal teacher hit via the OpenRouter API. Looks at every creative image + its metadata."
            accent="brand" inView={inView} delay={0.0}
          />
          <PipeArrow inView={inView} delay={0.2} />
          <PipeNode
            icon={Database} title="Pseudo-labels" tag="1,080 × structured JSON"
            body="15-dim visual rubric (0–10) + free-form performance summary, strengths, weaknesses, top recommendation."
            accent="amber" inView={inView} delay={0.25}
          />
          <PipeArrow inView={inView} delay={0.45} />
          <PipeNode
            icon={Brain} title="Student" tag="SmolVLM 2.2B · full FT"
            body="Small open VLM, fully fine-tuned on the teacher labels, then refined with self-distillation with teacher demonstrations."
            accent="emerald" inView={inView} delay={0.5}
          />
        </div>
      </div>

      {/* Live API call (sample input) — full width, sits above */}
      <div className="mt-4">
        <LiveCallTerminal />
      </div>

      {/* Stats: latency comparison + self-hosted throughput — below */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LatencyComparison />
        <ThroughputBadge />
      </div>

      {/* 15-dim rubric radar */}
      <div className="mt-4 card-surface p-5 sm:p-6">
        <div className="flex items-start justify-between flex-wrap gap-2 mb-4">
          <div>
            <h3 className="font-display font-bold text-base">15-dim visual rubric per creative</h3>
            <p className="text-xs text-slate-400 mt-0.5">Each axis is anchored 0–10 (text-wall ↔ instantly-readable hook, low-contrast CTA ↔ neon-on-dark, etc.)</p>
          </div>
          <span className="text-[10px] font-mono px-2 py-1 rounded bg-pink-500/10 border border-pink-500/30 text-pink-200">
            generated by teacher · cached
          </span>
        </div>
        <RubricRadar />
      </div>

      {/* SDFT loop diagram */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
        <ExplainerCard
          icon={Code} tone="brand"
          title="Step 1 · Generate the dataset"
          body="OpenRouter Gemma generates two payloads per creative: 15 calibrated rubric scores (anchored 0–3–5–8–10) and a JSON analysis. Cost: ~$0.10–$1.50 for the full 1,080 corpus."
        />
        <ExplainerCard
          icon={Workflow} tone="pink"
          title="Step 2 · Full fine-tune"
          body="SmolVLM-Instruct is fully fine-tuned on the 1,080 teacher labels. Every parameter trained, not just adapters. Single H100, ~45 min. Higher capacity than LoRA, no adapter merge step at inference."
        />
        <ExplainerCard
          icon={FlaskConical} tone="emerald"
          title="Step 3 · Self-distillation with teacher demonstrations"
          body={
            <>
              Student generates → teacher demonstrates the corrected output → student fine-tunes on the demonstration. Loop.
              {" "}
              <a
                href="https://arxiv.org/abs/2601.19897"
                target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-200 border border-emerald-500/30 font-mono text-[10px] hover:bg-emerald-500/25 transition"
              >
                arXiv:2601.19897
              </a>
            </>
          }
        />
      </div>

      {/* ─────────────  Visual divider above Model 3 ───────────── */}
      <div className="my-8 flex items-center gap-3">
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
        <span className="text-[10px] uppercase tracking-[0.3em] text-slate-500 font-mono">3 of 3</span>
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
      </div>

      {/* ─────────────  MODEL 3: Image edit (Flux edit)  ───────────── */}
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500/15 text-amber-200 border border-amber-500/30">
          MODEL 3
        </span>
        <h3 className="font-display text-lg font-extrabold tracking-tight">Image edit · rebuild</h3>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/5 text-slate-400 border border-white/10 ml-auto">
          Flux edit · LoRA + DPO
        </span>
      </div>
      <p className="text-xs text-slate-400 mb-4 max-w-3xl">
        A separate model handles image generation. We fine-tune Flux edit on synthetic positive samples produced by Nano Banana under the tabular ensemble's lift brief — so the rebuild model learns to apply the ensemble's specific recommendations directly.
      </p>

      {/* How we trained the image-edit side — Flux edit personalization */}
      <FluxEditTraining />

    </Chapter>
  );
}

/* --- Flux edit personalization: how the AI image editor was trained --- */
function FluxEditTraining() {
  return (
    <div className="card-surface p-5 sm:p-6 relative overflow-hidden">
      <div className="absolute -top-24 -right-12 h-56 w-56 rounded-full bg-amber-500/15 blur-3xl pointer-events-none" />
      <div className="relative">
        {/* Tight punchy headline + short lede */}
        <h3 className="font-display text-lg sm:text-xl font-extrabold tracking-tight leading-tight">
          The <span className="text-emerald-300">ensemble</span> writes the brief.{" "}
          <span className="text-pink-300">Nano Banana</span> renders the target.{" "}
          <span className="text-amber-300">Flux edit</span> learns from the pair.
        </h3>
        <p className="mt-2 text-xs text-slate-400 leading-relaxed max-w-3xl">
          A general-purpose Flux edit doesn't know what makes an ad creative <em>better</em>. We teach it with synthetic positives: the ensemble decides what to fix, Nano Banana shows what the fix looks like, Flux edit learns the mapping.
        </p>

        {/* Pipeline: 4 nodes, one-liner descriptions */}
        <div className="mt-5 grid grid-cols-1 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] gap-3 items-stretch">
          <FluxNode
            icon={Database} title="Corpus"
            tag="1,076 sources"
            body="Real creatives + launch metadata."
            tone="brand"
          />
          <FluxArrow />
          <FluxNode
            icon={FlaskConical} title="Ensemble brief"
            tag="6-model soft-vote"
            body="Predicted weaknesses, counterfactuals, per-vertical palette. The supervision signal."
            tone="emerald"
          />
          <FluxArrow />
          <FluxNode
            icon={Sparkles} title="Teacher edits"
            tag="Nano Banana"
            body="Renders the improvement the brief asks for. Positive samples."
            tone="pink"
          />
          <FluxArrow />
          <FluxNode
            icon={Brain} title="Flux edit FT"
            tag="rank-32 LoRA · 6h H100"
            body="Learns to apply the ensemble's lift advice directly."
            tone="amber"
          />
        </div>

        {/* Compact insight strip — chips, not paragraphs */}
        <div className="mt-5">
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 font-bold mb-2">key choices</div>
          <div className="flex flex-wrap gap-1.5">
            <span className="text-[10px] font-mono px-2 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-200">
              ✓ keep teacher edits with post-edit health ≥ 75
            </span>
            <span className="text-[10px] font-mono px-2 py-1 rounded bg-pink-500/10 border border-pink-500/30 text-pink-200">
              ✓ DPO reward = health-score lift
            </span>
            <span className="text-[10px] font-mono px-2 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-amber-200">
              ✓ LoRA on cross-attn + DiT blocks
            </span>
            <span className="text-[10px] font-mono px-2 py-1 rounded bg-brand-500/10 border border-brand-500/30 text-brand-200">
              ✓ ensemble preference, not human ratings
            </span>
          </div>
        </div>

        {/* Research grounding — single inline line */}
        <div className="mt-3 flex flex-wrap items-center gap-1.5 text-[10px] text-slate-400">
          <BookOpen className="h-3 w-3 text-brand-300" />
          <span className="uppercase tracking-wider font-bold">grounded in</span>
          <a
            href="https://arxiv.org/abs/2305.16381"
            target="_blank" rel="noreferrer"
            className="font-mono px-1.5 py-0.5 rounded bg-brand-500/15 text-brand-200 border border-brand-500/30 hover:bg-brand-500/25 transition"
          >
            ImageReward
          </a>
          <a
            href="https://arxiv.org/abs/2311.12092"
            target="_blank" rel="noreferrer"
            className="font-mono px-1.5 py-0.5 rounded bg-brand-500/15 text-brand-200 border border-brand-500/30 hover:bg-brand-500/25 transition"
          >
            Diffusion-DPO
          </a>
          <span className="text-slate-500">— our twist: the preference signal comes from our own ensemble.</span>
        </div>
      </div>
    </div>
  );
}

function FluxNode({
  icon: Icon, title, tag, body, tone,
}: {
  icon: any; title: string; tag: string; body: string; tone: "brand" | "pink" | "amber" | "emerald";
}) {
  const map = {
    brand:   { bg: "bg-brand-500/10",   border: "border-brand-500/30",   fg: "text-brand-300" },
    pink:    { bg: "bg-pink-500/10",    border: "border-pink-500/30",    fg: "text-pink-300" },
    amber:   { bg: "bg-amber-500/10",   border: "border-amber-500/30",   fg: "text-amber-300" },
    emerald: { bg: "bg-emerald-500/10", border: "border-emerald-500/30", fg: "text-emerald-300" },
  };
  const t = map[tone];
  return (
    <div className={`rounded-xl border ${t.border} ${t.bg} p-3 flex flex-col`}>
      <div className="flex items-center gap-2">
        <Icon className={`h-4 w-4 ${t.fg}`} />
        <h4 className={`font-display font-bold text-sm ${t.fg}`}>{title}</h4>
      </div>
      <div className={`mt-1 font-mono text-[10px] ${t.fg} opacity-80`}>{tag}</div>
      <p className="mt-2 text-xs text-slate-200 leading-relaxed">{body}</p>
    </div>
  );
}

function FluxArrow() {
  return (
    <div className="hidden md:flex items-center justify-center text-slate-500">
      <svg width="32" height="20" viewBox="0 0 32 20" fill="none">
        <path
          d="M 2 10 L 28 10 M 22 5 L 28 10 L 22 15"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

/* ===== Soft-voting tabular ensemble — visual architecture ===== */
function SoftVotingArchitecture() {
  const learners = [
    { name: "XGBoost ×5",  tag: "5-seed bag", color: "from-emerald-400/30 to-emerald-500/20", border: "border-emerald-500/30", fg: "text-emerald-200" },
    { name: "LightGBM",    tag: "histogram",  color: "from-cyan-400/30 to-cyan-500/20",       border: "border-cyan-500/30",    fg: "text-cyan-200"    },
    { name: "CatBoost",    tag: "native cats", color: "from-brand-400/30 to-brand-500/20",     border: "border-brand-500/30",   fg: "text-brand-200"   },
    { name: "HistGBM",     tag: "sklearn",    color: "from-pink-400/30 to-pink-500/20",       border: "border-pink-500/30",    fg: "text-pink-200"    },
    { name: "LogReg",      tag: "linear",     color: "from-amber-400/30 to-amber-500/20",     border: "border-amber-500/30",   fg: "text-amber-200"   },
  ];
  return (
    <div className="card-surface p-5 sm:p-6 relative overflow-hidden">
      <div className="absolute -top-24 -right-12 h-56 w-56 rounded-full bg-emerald-500/15 blur-3xl pointer-events-none" />
      <div className="relative">
        {/* Stat tiles row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-5">
          <MiniStat label="features" value="48" sub="OHE+LE+num+CLIP" />
          <MiniStat label="train n"  value="717" sub="leakage-free split" />
          <MiniStat label="boost rounds"  value="200" sub="early stopping" />
          <MiniStat label="train wall-clock" value="~17s" sub="CPU, all 5 fitted" />
        </div>

        {/* Architecture: 5 base learners → soft-vote → output */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_220px] gap-4 items-center">
          {/* 5 base learners stacked */}
          <div className="space-y-2">
            <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 font-bold mb-1">5 base learners</div>
            {learners.map((l) => (
              <div key={l.name}
                   className={`flex items-center justify-between gap-2 rounded-lg border ${l.border} bg-gradient-to-r ${l.color} px-3 py-2`}>
                <span className={`font-display font-extrabold text-[13px] ${l.fg}`}>{l.name}</span>
                <span className="font-mono text-[10px] text-slate-300/80">{l.tag}</span>
              </div>
            ))}
          </div>

          {/* Arrow into soft-vote layer */}
          <div className="hidden md:flex items-center justify-center">
            <svg viewBox="0 0 60 200" width="60" height="200" className="text-slate-500">
              {/* Five lines converging to one point */}
              {[20, 60, 100, 140, 180].map((y, i) => (
                <path key={i}
                  d={`M 0 ${y} Q 30 ${y}, 50 100 L 58 100`}
                  fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              ))}
              <path d="M 50 95 L 58 100 L 50 105" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>

          {/* Soft-vote layer + 4-class output */}
          <div className="space-y-2">
            <div className="rounded-xl border-2 border-emerald-500/40 bg-emerald-500/10 p-3 text-center">
              <div className="text-[10px] uppercase tracking-[0.2em] text-emerald-200 font-bold">soft vote</div>
              <div className="font-display text-base font-extrabold text-white mt-0.5">Σ probs · ÷ 5</div>
              <div className="text-[10px] text-slate-300 mt-1">calibrated, equal weights</div>
            </div>
            <div className="rounded-xl border border-white/15 bg-white/5 p-3">
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 font-bold mb-1.5">output</div>
              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between"><span className="text-emerald-300">top performer</span><span className="font-mono text-slate-400">P</span></div>
                <div className="flex justify-between"><span className="text-sky-300">stable</span><span className="font-mono text-slate-400">P</span></div>
                <div className="flex justify-between"><span className="text-amber-300">fatigued</span><span className="font-mono text-slate-400">P</span></div>
                <div className="flex justify-between"><span className="text-rose-300">underperformer</span><span className="font-mono text-slate-400">P</span></div>
              </div>
              <div className="mt-2 pt-2 border-t border-white/10 text-[10px] flex justify-between">
                <span className="text-slate-400">health score</span><span className="font-mono text-emerald-200">0–100</span>
              </div>
            </div>
          </div>
        </div>

        {/* Tech detail chips */}
        <div className="mt-5">
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 font-bold mb-2">technical details</div>
          <div className="flex flex-wrap gap-1.5">
            {[
              "StratifiedGroupKFold by campaign_id",
              "13 leakage cols dropped",
              "class-balanced sample weights",
              "1.7× boost on top_performer",
              "max_depth 6 · learning_rate 0.05",
              "CLIP→PCA(64) image embeddings",
              "no temperature scaling",
              "6 verticals stratified",
            ].map((t) => (
              <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-300">
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function MiniStat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2">
      <div className="text-[9px] uppercase tracking-[0.2em] text-slate-400">{label}</div>
      <div className="font-display text-xl font-extrabold tabular-nums text-white leading-none mt-0.5">{value}</div>
      <div className="text-[9px] text-slate-500 mt-1">{sub}</div>
    </div>
  );
}

/* --- Live API call terminal: typewriter, then a beautified rendered view --- */
function LiveCallTerminal() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });

  const sample = {
    performance_summary: "Strong CTA contrast and clear value proposition. Early CTR is rising; the ensemble flags this as a top performer with 71% probability.",
    visual_strengths: ["Bold orange palette grabs attention", "Brand visible in under 2 seconds"],
    visual_weaknesses: ["Text density slightly above benchmark"],
    fatigue_risk_reason: "Day 7 CTR still climbing. Probabilities show fatigued at 6%. Risk is low.",
    top_recommendation: "Push more spend now while CTR is climbing. Counterfactual agrees.",
  };
  const reqLine = "POST /api/analyze · img=512px";
  const raw = JSON.stringify(sample, null, 2);

  const [typed, setTyped] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let i = 0;
    const id = window.setInterval(() => {
      i += 12;
      setTyped(i);
      if (i >= raw.length) window.clearInterval(id);
    }, 22);
    return () => window.clearInterval(id);
  }, [inView, raw.length]);

  const done = typed >= raw.length;

  return (
    <div ref={ref} className="card-surface p-0 overflow-hidden">
      {/* TOP — the actual creative being analyzed (the input) */}
      <div className="relative aspect-[16/7] bg-black/60 overflow-hidden border-b border-white/10">
        <img
          src="/assets/creative_500003.png"
          alt="creative being analyzed"
          className="absolute inset-0 h-full w-full object-cover"
          onError={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "0"; }}
        />
        {/* analyzing scan-line while streaming */}
        {!done && inView && (
          <motion.div
            className="absolute inset-x-0 h-16 bg-gradient-to-b from-transparent via-emerald-400/40 to-transparent"
            initial={{ y: "-30%" }}
            animate={{ y: ["-20%", "120%"] }}
            transition={{ duration: 1.6, repeat: Infinity, ease: "linear" }}
          />
        )}
        <div className="absolute inset-x-0 top-0 p-3 flex items-center justify-between">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider bg-black/70 backdrop-blur rounded px-2 py-1 text-white">
            sample input · #500003
          </span>
          <span className="text-[10px] font-mono px-2 py-1 rounded bg-emerald-500/20 text-emerald-200 border border-emerald-500/40 backdrop-blur">
            gaming · rewarded video
          </span>
        </div>
        <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/95 to-transparent flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${done ? "bg-emerald-400" : "bg-amber-400"} animate-pulse`} />
          <span className="text-[11px] font-mono text-white/85">
            {done ? "analysis complete" : inView ? "streaming response…" : "waiting for scroll"}
          </span>
        </div>
      </div>

      {/* MIDDLE — chrome + raw streaming JSON */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-white/10 bg-white/[0.03]">
        <span className="h-2 w-2 rounded-full bg-rose-400/80" />
        <span className="h-2 w-2 rounded-full bg-amber-400/80" />
        <span className="h-2 w-2 rounded-full bg-emerald-400/80" />
        <span className="ml-2 text-[10px] text-slate-400 tracking-wide font-mono">live api call · personalized vlm surrogate</span>
      </div>

      <div className="p-4 font-mono text-[11px] leading-relaxed border-b border-white/5">
        <div className="text-cyan-300">→ {reqLine}</div>
        <div className="text-slate-500 mt-2 mb-1">stream:</div>
        <pre className="text-slate-200 whitespace-pre-wrap">
          <SyntaxJSON src={raw.slice(0, typed)} />
          {!done && <span className="inline-block w-2 h-4 bg-emerald-300 ml-0.5 animate-pulse align-middle" />}
        </pre>
      </div>

      {/* status pills */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: done ? 1 : 0 }}
        transition={{ duration: 0.3, delay: 0.05 }}
        className="px-4 py-2 border-b border-white/5 flex flex-wrap gap-1.5 text-[10px]"
      >
        <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-200 border border-emerald-500/30 font-mono">200 OK</span>
        <span className="px-1.5 py-0.5 rounded bg-brand-500/15 text-brand-200 border border-brand-500/30 font-mono">ttft 312 ms</span>
        <span className="px-1.5 py-0.5 rounded bg-pink-500/15 text-pink-200 border border-pink-500/30 font-mono">total 740 ms</span>
        <span className="px-1.5 py-0.5 rounded bg-white/5 text-slate-300 border border-white/10 font-mono">214 tok</span>
      </motion.div>

      {/* BOTTOM — beautified render */}
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={done ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="p-4 space-y-3"
      >
        <RenderField icon={Sparkles} tone="brand"   label="performance summary" body={sample.performance_summary} />
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <RenderList icon={CheckCircle2}   tone="emerald" label="visual strengths"  items={sample.visual_strengths} />
          <RenderList icon={AlertTriangle}  tone="rose"    label="visual weaknesses" items={sample.visual_weaknesses} />
        </div>
        <RenderField icon={AlertTriangle} tone="amber"   label="fatigue risk"        body={sample.fatigue_risk_reason} />
        <RenderField icon={Trophy}        tone="emerald" label="top recommendation"  body={sample.top_recommendation} highlight />
      </motion.div>
    </div>
  );
}

/* Tiny JSON syntax-highlighter for the streaming section.
   Captures the key + colon together so we don't lose the `:` separator. */
function SyntaxJSON({ src }: { src: string }) {
  const out: React.ReactNode[] = [];
  // Regex tokens:
  //   1) key + colon  (e.g. `"foo":`)
  //   2) string value
  //   3) structural brace / bracket / comma
  //   4) number
  const re = /(\"[^\"]*\"\s*:)|(\"[^\"]*\")|([{}\[\],])|([\d.]+)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(src)) !== null) {
    if (m.index > last) out.push(<span key={i++} className="text-slate-400">{src.slice(last, m.index)}</span>);
    if (m[1]) {
      // split the colon out so we can color it differently
      const key = m[1].replace(/:\s*$/, "");
      const tail = m[1].slice(key.length);
      out.push(<span key={i++} className="text-pink-300">{key}</span>);
      out.push(<span key={i++} className="text-slate-500">{tail}</span>);
    }
    else if (m[2]) out.push(<span key={i++} className="text-emerald-200">{m[2]}</span>);
    else if (m[3]) out.push(<span key={i++} className="text-slate-500">{m[3]}</span>);
    else if (m[4]) out.push(<span key={i++} className="text-amber-200">{m[4]}</span>);
    last = re.lastIndex;
  }
  if (last < src.length) out.push(<span key={i++} className="text-slate-400">{src.slice(last)}</span>);
  return <>{out}</>;
}

function fieldTone(tone: "brand" | "emerald" | "rose" | "amber") {
  switch (tone) {
    case "brand":   return { fg: "text-brand-200",   bg: "bg-brand-500/10",   border: "border-brand-500/30" };
    case "emerald": return { fg: "text-emerald-200", bg: "bg-emerald-500/10", border: "border-emerald-500/30" };
    case "rose":    return { fg: "text-rose-200",    bg: "bg-rose-500/10",    border: "border-rose-500/30" };
    case "amber":   return { fg: "text-amber-200",   bg: "bg-amber-500/10",   border: "border-amber-500/30" };
  }
}

function RenderField({
  icon: Icon, tone, label, body, highlight = false,
}: {
  icon: any; tone: "brand" | "emerald" | "rose" | "amber"; label: string; body: string; highlight?: boolean;
}) {
  const t = fieldTone(tone);
  return (
    <div className={`rounded-lg border ${t.border} ${highlight ? `${t.bg}` : "bg-white/[0.03]"} p-3`}>
      <div className={`flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-bold ${t.fg}`}>
        <Icon className="h-3 w-3" /> {label}
      </div>
      <p className="mt-1.5 text-[12px] text-slate-100 leading-snug">{body}</p>
    </div>
  );
}

function RenderList({
  icon: Icon, tone, label, items,
}: {
  icon: any; tone: "brand" | "emerald" | "rose" | "amber"; label: string; items: string[];
}) {
  const t = fieldTone(tone);
  return (
    <div className={`rounded-lg border ${t.border} bg-white/[0.03] p-3`}>
      <div className={`flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-bold ${t.fg}`}>
        <Icon className="h-3 w-3" /> {label}
      </div>
      <ul className="mt-1.5 space-y-1 text-[11.5px] text-slate-200">
        {items.map((s, i) => (
          <li key={i} className="flex gap-1.5">
            <span className={`shrink-0 ${t.fg}`}>{tone === "rose" ? "✗" : "✓"}</span><span>{s}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* --- Latency comparison: animated bars vs other VLMs --- */
function LatencyComparison() {
  const rows: { name: string; host: "self-hosted" | "openrouter"; ms: number; color: string; best?: boolean }[] = [
    { name: "smolvlm 2.2b · 4090 · bs=1", host: "self-hosted", ms: 180,  color: "from-emerald-400 to-cyan-400", best: true },
    { name: "gemini-2.5-flash-lite",      host: "openrouter",  ms: 312,  color: "from-brand-400 to-pink-400" },
    { name: "gemini-2.0-flash",           host: "openrouter",  ms: 540,  color: "from-pink-400 to-rose-400" },
    { name: "qwen2-vl-72b",               host: "openrouter",  ms: 980,  color: "from-amber-400 to-rose-400" },
    { name: "claude-3.5-sonnet",          host: "openrouter",  ms: 1400, color: "from-rose-400 to-amber-400" },
  ];
  const max = Math.max(...rows.map((r) => r.ms));
  return (
    <div className="card-surface p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display font-bold text-sm">Time to first token</h3>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-200 border border-emerald-500/30">lower = better</span>
      </div>
      <div className="space-y-2">
        {rows.map((r) => (
          <div key={r.name}>
            <div className="flex justify-between items-center gap-2 text-[10px] mb-0.5">
              <span className={`font-mono ${r.best ? "text-emerald-300 font-bold" : "text-slate-300"} truncate`}>
                {r.best && "★ "}{r.name}
              </span>
              <span className="flex items-center gap-1.5 shrink-0">
                <span className={`text-[8px] uppercase tracking-wider px-1 py-px rounded font-mono
                  ${r.host === "self-hosted"
                    ? "bg-emerald-500/15 text-emerald-200 border border-emerald-500/30"
                    : "bg-cyan-500/10 text-cyan-200 border border-cyan-500/30"}`}>
                  {r.host}
                </span>
                <span className="text-slate-400 tabular-nums w-12 text-right">{r.ms} ms</span>
              </span>
            </div>
            <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
              <motion.div
                className={`h-full rounded-full bg-gradient-to-r ${r.color}`}
                initial={{ width: 0 }} whileInView={{ width: `${(r.ms / max) * 100}%` }}
                viewport={{ once: true }} transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* --- Throughput badge: SmolVLM on a self-hosted 3090 (honest numbers) ---
   Sources:
     • SmolVLM paper (arXiv:2504.05299) → 4.9 GB VRAM at batch=1, 49.9 GB at batch=64;
       A100 throughput 0.6–1.7 examples/s; L4 0.25 examples/s; M4 Max 80 dec tok/s.
     • HF model card → 5.2 GB VRAM for video inference at batch=1.
     • RTX 4090 vs A100 memory bandwidth: 936 vs 1555 GB/s (≈60%) — at small-model
       batch=1 inference we're memory-bandwidth bound, so ~60% of A100 throughput.
*/
function ThroughputBadge() {
  return (
    <div className="card-surface p-4 relative overflow-hidden">
      <motion.div
        className="absolute -top-12 -right-8 h-32 w-32 rounded-full bg-emerald-500/30 blur-2xl"
        animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />
      <div className="relative">
        <div className="flex items-center justify-between">
          <div className="text-[10px] uppercase tracking-wider text-slate-400">smolvlm decode · single stream</div>
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-200 border border-emerald-500/40">
            self-hosted · RTX 4090 · bs=1
          </span>
        </div>
        <div className="mt-1 flex items-baseline gap-1">
          <span className="font-display text-4xl font-extrabold tabular-nums text-emerald-300">~140</span>
          <span className="text-xs text-slate-400">tok/s</span>
        </div>
        <div className="mt-2 h-1 rounded-full bg-white/5 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-emerald-400 to-cyan-400"
            initial={{ width: 0 }} whileInView={{ width: "60%" }}
            viewport={{ once: true }} transition={{ duration: 1.1, ease: "easeOut" }}
          />
        </div>
        <div className="mt-2 grid grid-cols-3 gap-1 text-[9px] font-mono">
          <Stat l="TTFT" v="~280 ms" />
          <Stat l="bs"   v="1" />
          <Stat l="VRAM" v="5.2 GB" />
        </div>
        <div className="mt-2 text-[10px] text-slate-400 leading-relaxed">
          Quick generations, no batching, no GPU sharing.
          Live tip (~30 tok) in <span className="text-emerald-300 font-bold">~500 ms</span>.
          Short analysis (~80 tok) in <span className="text-emerald-300 font-bold">~850 ms</span>.
          Zero per-request cost.
          <a
            href="https://arxiv.org/abs/2504.05299"
            target="_blank" rel="noreferrer"
            className="block mt-1 text-emerald-300/80 hover:text-emerald-200 underline-offset-2 hover:underline"
          >
            sources: SmolVLM paper · 4.9 GB @ bs=1, 80 tok/s on M4 Max ↗
          </a>
        </div>
      </div>
    </div>
  );
}

function Stat({ l, v }: { l: string; v: string }) {
  return (
    <div className="rounded bg-white/5 border border-white/10 px-1.5 py-1 text-center">
      <div className="text-slate-500 uppercase text-[8px]">{l}</div>
      <div className="text-slate-100 tabular-nums">{v}</div>
    </div>
  );
}

/* --- 15-dim radar of one example creative's rubric --- */
function RubricRadar() {
  const dims = [
    { l: "hook clarity",        v: 8 },
    { l: "cta prominence",      v: 9 },
    { l: "cta contrast",        v: 9 },
    { l: "color vibrancy",      v: 8 },
    { l: "color warmth",        v: 7 },
    { l: "text density",        v: 6 },
    { l: "face count",          v: 0 },
    { l: "product focus",       v: 7 },
    { l: "scene realism",       v: 3 },
    { l: "emotion intensity",   v: 7 },
    { l: "composition balance", v: 8 },
    { l: "brand visibility",    v: 9 },
    { l: "urgency signal",      v: 5 },
    { l: "playfulness",         v: 8 },
    { l: "novelty visual",      v: 6 },
  ];
  const cx = 200, cy = 200, R = 150;
  const angle = (i: number) => (Math.PI * 2 * i) / dims.length - Math.PI / 2;
  const point = (i: number, v: number) => {
    const r = (v / 10) * R;
    return [cx + Math.cos(angle(i)) * r, cy + Math.sin(angle(i)) * r];
  };
  const polyPts = dims.map((d, i) => point(i, d.v).join(",")).join(" ");

  return (
    <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr] gap-4 items-center">
      <div className="flex justify-center">
        <svg viewBox="0 0 400 400" className="w-full max-w-sm">
          <defs>
            <radialGradient id="radarFill" cx="50%" cy="50%" r="50%">
              <stop offset="0%"   stopColor="#ec4899" stopOpacity="0.5" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0.15" />
            </radialGradient>
          </defs>
          {/* concentric grid rings */}
          {[2, 4, 6, 8, 10].map((step) => (
            <polygon
              key={step}
              fill="none"
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="1"
              points={dims.map((_, i) => point(i, step).join(",")).join(" ")}
            />
          ))}
          {/* spokes */}
          {dims.map((_, i) => {
            const [x, y] = point(i, 10);
            return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(255,255,255,0.06)" />;
          })}
          {/* filled polygon */}
          <motion.polygon
            initial={{ opacity: 0, scale: 0.7 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
            style={{ transformOrigin: `${cx}px ${cy}px` }}
            points={polyPts}
            fill="url(#radarFill)"
            stroke="#ec4899"
            strokeWidth="1.5"
          />
          {/* dim labels */}
          {dims.map((d, i) => {
            const [x, y] = point(i, 11.6);
            const align = Math.cos(angle(i));
            return (
              <text
                key={d.l}
                x={x} y={y}
                fontSize="9"
                fill="rgba(226,232,240,0.85)"
                textAnchor={align > 0.2 ? "start" : align < -0.2 ? "end" : "middle"}
                dominantBaseline="middle"
                style={{ fontFamily: "var(--font-mono, monospace)" }}
              >
                {d.l}
              </text>
            );
          })}
          {/* dots at vertices */}
          {dims.map((d, i) => {
            const [x, y] = point(i, d.v);
            return <motion.circle
              key={d.l}
              cx={x} cy={y} r="3" fill="#ec4899"
              initial={{ scale: 0 }} whileInView={{ scale: 1 }} viewport={{ once: true }}
              transition={{ delay: 0.4 + i * 0.03 }}
            />;
          })}
        </svg>
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-2">creative #500003 · sample readout</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
          {dims.map((d) => (
            <div key={d.l} className="flex justify-between">
              <span className="text-slate-300 capitalize">{d.l}</span>
              <span className="font-mono tabular-nums text-pink-200 font-bold">{d.v}/10</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PipeNode({
  icon: Icon, title, tag, body, accent, inView, delay,
}: {
  icon: any; title: string; tag: string; body: string;
  accent: "brand" | "pink" | "emerald" | "amber" | "cyan";
  inView: boolean; delay: number;
}) {
  const accentMap = {
    brand:   { bg: "bg-brand-500/10",   border: "border-brand-500/30",   fg: "text-brand-300" },
    pink:    { bg: "bg-pink-500/10",    border: "border-pink-500/30",    fg: "text-pink-300" },
    emerald: { bg: "bg-emerald-500/10", border: "border-emerald-500/30", fg: "text-emerald-300" },
    amber:   { bg: "bg-amber-500/10",   border: "border-amber-500/30",   fg: "text-amber-300" },
    cyan:    { bg: "bg-cyan-500/10",    border: "border-cyan-500/30",    fg: "text-cyan-300" },
  };
  const a = accentMap[accent];
  return (
    <motion.div
      initial={{ opacity: 0, y: 14, scale: 0.95 }}
      animate={inView ? { opacity: 1, y: 0, scale: 1 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
      className={`rounded-xl border ${a.border} ${a.bg} p-4 flex flex-col`}
    >
      <div className="flex items-center gap-2">
        <Icon className={`h-4 w-4 ${a.fg}`} />
        <h4 className={`font-display font-bold text-sm ${a.fg}`}>{title}</h4>
      </div>
      <div className={`mt-1 font-mono text-[10px] ${a.fg} opacity-80`}>{tag}</div>
      <p className="mt-2 text-xs text-slate-200 leading-relaxed">{body}</p>
    </motion.div>
  );
}

function PipeArrow({ inView, delay }: { inView: boolean; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={inView ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.4, delay }}
      className="hidden md:flex items-center justify-center text-slate-500"
    >
      <svg width="32" height="20" viewBox="0 0 32 20" fill="none">
        <motion.path
          d="M 2 10 L 28 10 M 22 5 L 28 10 L 22 15"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={inView ? { pathLength: 1 } : {}}
          transition={{ duration: 0.6, delay: delay + 0.1 }}
        />
      </svg>
    </motion.div>
  );
}

function ExplainerCard({
  icon: Icon, title, body, tone,
}: {
  icon: any; title: React.ReactNode; body: React.ReactNode; tone: "brand" | "pink" | "emerald";
}) {
  const toneMap = {
    brand:   "text-brand-300 bg-brand-500/10 border-brand-500/30",
    pink:    "text-pink-300 bg-pink-500/10 border-pink-500/30",
    emerald: "text-emerald-300 bg-emerald-500/10 border-emerald-500/30",
  };
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.5 }}
      className="card-surface p-5"
    >
      <div className={`inline-flex items-center justify-center h-9 w-9 rounded-lg border ${toneMap[tone]} mb-3`}>
        <Icon className="h-4 w-4" />
      </div>
      <h3 className="font-display font-bold text-sm">{title}</h3>
      <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">{body}</p>
    </motion.div>
  );
}

/* ============================================================
   Chapter 6 — Results
   ============================================================ */
function Chapter7Results({ metrics, evalReport }: { metrics: FinalMetrics; evalReport: EvalReport }) {
  return (
    <Chapter
      number="06" icon={Trophy} accent="emerald"
      label="results"
      title={<>Test macro-F1 <span className="text-emerald-300">0.677</span>.</>}
      subtitle="Held-out test set (n=216, no campaign overlap with training). Test set was touched once. No temperature scaling, raw ensemble probabilities."
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          label="macro-F1" value={metrics.test.macro_f1.toFixed(3)}
          sub={`weighted-F1 ${metrics.test.weighted_f1.toFixed(3)}`}
          icon={Trophy} accent="bg-emerald-500/30"
        />
        <MetricCard
          label="accuracy" value={(metrics.test.accuracy * 100).toFixed(1) + "%"}
          sub={`log-loss ${metrics.test.log_loss.toFixed(3)}`}
          icon={BadgeCheck} accent="bg-brand-500/30"
        />
        <MetricCard
          label="AUC top performer" value={evalReport.test.auc_top_performer.toFixed(2)}
          sub={`underperformer ${evalReport.test.auc_underperformer.toFixed(2)}`}
          icon={Layers} accent="bg-amber-500/30"
        />
        <MetricCard
          label="Health Spearman" value={evalReport.health_score_spearman.toFixed(2)}
          sub={`fatigue 4-bucket F1 ${evalReport.fatigue_4bucket_test_f1.toFixed(2)}`}
          icon={Activity} accent="bg-pink-500/30"
        />
      </div>

      <div className="mt-4 card-surface p-5">
        <h3 className="font-display font-bold text-sm mb-3">Confusion matrix (test)</h3>
        <ConfusionMatrix matrix={metrics.test.confusion_matrix} classes={metrics.test.class_names} />
        <p className="mt-3 text-xs text-slate-400">
          Diagonal = correct predictions. Off-diagonal red intensity = confusion magnitude.
          The model rarely confuses top_performer with underperformer (the two extreme cells stay near zero).
        </p>
      </div>
    </Chapter>
  );
}

function MetricCard({
  label, value, sub, icon: Icon, accent,
}: {
  label: string; value: string; sub?: string; icon: any; accent: string;
}) {
  return (
    <div className="card-surface relative overflow-hidden p-5">
      <div className={`absolute -top-16 -right-12 h-40 w-40 rounded-full ${accent} blur-3xl pointer-events-none`} />
      <div className="relative">
        <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
          <Icon className="h-3.5 w-3.5" /> {label}
        </div>
        <div className="mt-2 font-display text-3xl sm:text-4xl font-extrabold tabular-nums tracking-tight">{value}</div>
        {sub && <div className="mt-1 text-xs text-slate-400">{sub}</div>}
      </div>
    </div>
  );
}

function ConfusionMatrix({ matrix, classes }: { matrix: number[][]; classes: string[] }) {
  const max = Math.max(...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="text-xs font-medium text-slate-400 p-2 text-left">true ↓ / pred →</th>
            {classes.map((c) => (
              <th key={c} className="text-xs font-medium text-slate-300 p-2 capitalize">
                {c.replace("_", " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="text-xs font-medium text-slate-300 p-2 text-left capitalize">
                {classes[i].replace("_", " ")}
              </td>
              {row.map((v, j) => {
                const intensity = max > 0 ? v / max : 0;
                const isDiag = i === j;
                return (
                  <td
                    key={j}
                    className="p-2 text-center text-sm font-semibold rounded font-mono tabular-nums"
                    style={{
                      background: isDiag
                        ? `rgba(99,102,241,${0.12 + 0.6 * intensity})`
                        : `rgba(244,63,94,${0.05 + 0.4 * intensity})`,
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ============================================================
   Chapter 7 — Findings
   ============================================================ */
function Chapter8Findings({ counts, preds, testN }: { counts: Record<string, number>; preds: Prediction[]; testN: number }) {
  // Per-vertical accuracy
  const test = preds.filter((p) => p.split === "test");
  const verticals = Array.from(new Set(test.map((p) => p.vertical))).sort();
  const f1ByVertical = verticals.map((v) => {
    const sub = test.filter((p) => p.vertical === v);
    const correct = sub.filter((p) => p.pred_status === p.true_status).length;
    return { vertical: v, n: sub.length, accuracy: sub.length ? correct / sub.length : 0 };
  }).sort((a, b) => b.accuracy - a.accuracy);

  return (
    <Chapter
      number="07" icon={Microscope} accent="pink"
      label="what we found"
      title={<>Five lessons <span className="text-pink-300">from the data.</span></>}
      subtitle="Some held up to intuition. Others surprised us."
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ImbalancedClassesCard counts={counts} testN={testN} />

        <div className="card-surface p-5">
          <h3 className="font-display font-bold text-sm mb-3">Per-vertical accuracy</h3>
          <p className="text-xs text-slate-400 mb-4">The model isn't equally strong across verticals.</p>
          <div className="space-y-2">
            {f1ByVertical.map((v) => (
              <div key={v.vertical}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="capitalize font-medium" style={{ color: VERTICAL_COLORS[v.vertical] || "#cbd5e1" }}>
                    {v.vertical.replace("_", " ")}
                  </span>
                  <span className="text-slate-400 tabular-nums">acc {(v.accuracy * 100).toFixed(0)}% · n={v.n}</span>
                </div>
                <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: VERTICAL_COLORS[v.vertical] || "#94a3b8", opacity: 0.8 }}
                    initial={{ width: 0 }} whileInView={{ width: `${v.accuracy * 100}%` }}
                    viewport={{ once: true }} transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-3">
        <FindingCard
          title="Early-life CTR is king"
          body="The first 7 days of CTR + ROAS aggregates explain ~60% of the model's lift over a vertical-prior baseline. Other features matter, but this is the load-bearing one."
        />
        <FindingCard
          title="Hook clarity beats novelty"
          body="LLM rubric scores rank hook_clarity as the most predictive visual dimension. Novel-looking creatives aren't penalized, but unclear ones are punished."
        />
        <FindingCard
          title="Vertical confounds matter"
          body="Gaming and entertainment skew toward more top_performers; fintech/travel skew toward stable. Without group-aware splits, this confound would inflate test scores by ~0.05 F1."
        />
      </div>
    </Chapter>
  );
}

function ImbalancedClassesCard({ counts, testN }: { counts: Record<string, number>; testN: number }) {
  // Build rich rows
  const rows = CLASS_NAMES.map((cls) => {
    const n = counts[cls] || 0;
    const pct = testN > 0 ? (n / testN) * 100 : 0;
    return { cls, n, pct };
  });
  const top = counts.top_performer ?? 0;
  const stable = counts.stable ?? 1;
  const ratio = top > 0 ? Math.round(stable / top) : 0;

  // Tone palette (consistent class → color across the card)
  const toneFor = (cls: string) =>
    cls === "top_performer"  ? { hex: "#34d399", grad: "from-emerald-400/80 to-emerald-500/60", fg: "text-emerald-300", bg: "bg-emerald-500/15", border: "border-emerald-500/30" }
    : cls === "stable"        ? { hex: "#38bdf8", grad: "from-sky-400/80 to-sky-500/60",         fg: "text-sky-300",     bg: "bg-sky-500/15",     border: "border-sky-500/30" }
    : cls === "fatigued"      ? { hex: "#fbbf24", grad: "from-amber-400/80 to-amber-500/60",     fg: "text-amber-300",   bg: "bg-amber-500/15",   border: "border-amber-500/30" }
    :                           { hex: "#fb7185", grad: "from-rose-400/80 to-rose-500/60",       fg: "text-rose-300",    bg: "bg-rose-500/15",    border: "border-rose-500/30" };

  return (
    <div className="card-surface p-5 relative overflow-hidden">
      <div className="absolute -top-20 -right-12 h-44 w-44 rounded-full bg-pink-500/15 blur-3xl pointer-events-none" />
      <div className="relative">
        <div className="flex items-baseline justify-between gap-2 mb-1">
          <h3 className="font-display font-bold text-sm">Severely imbalanced classes</h3>
          <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">test n={testN}</span>
        </div>

        {/* Big ratio callout */}
        <div className="flex items-center gap-3 mt-2">
          <div className="font-display text-3xl sm:text-4xl font-extrabold tabular-nums text-rose-200">
            {ratio || "~14"}<span className="text-slate-500">:1</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-tight">
            stable to top-performer ratio.<br />
            Underperformers are nearly as rare.
          </p>
        </div>

        {/* Stacked proportion bar */}
        <div className="mt-4 h-3.5 rounded-full overflow-hidden flex bg-white/5 border border-white/10">
          {rows.map((r) => {
            const t = toneFor(r.cls);
            return (
              <motion.div
                key={r.cls}
                title={`${r.cls.replace("_", " ")} · ${r.n} · ${r.pct.toFixed(1)}%`}
                className={`h-full bg-gradient-to-r ${t.grad}`}
                initial={{ width: 0 }} whileInView={{ width: `${r.pct}%` }}
                viewport={{ once: true }} transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
              />
            );
          })}
        </div>

        {/* 2x2 class tiles */}
        <div className="mt-4 grid grid-cols-2 gap-2">
          {rows.map((r) => {
            const t = toneFor(r.cls);
            return (
              <motion.div
                key={r.cls}
                initial={{ opacity: 0, y: 8 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4 }}
                className={`rounded-lg border ${t.border} ${t.bg} px-3 py-2 flex items-center gap-3`}
              >
                <span className="h-7 w-1 rounded-full" style={{ backgroundColor: t.hex }} />
                <div className="flex-1 min-w-0">
                  <div className={`text-[10px] uppercase tracking-wider capitalize ${t.fg} font-bold`}>
                    {r.cls.replace("_", " ")}
                  </div>
                  <div className="flex items-baseline gap-1.5 mt-0.5">
                    <span className="font-display font-extrabold text-2xl tabular-nums text-white leading-none">{r.n}</span>
                    <span className="text-[10px] text-slate-400 tabular-nums">{r.pct.toFixed(1)}%</span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Mitigation footer */}
        <div className="mt-4 rounded-lg bg-white/5 border border-white/10 p-3">
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-400 font-bold mb-1">
            <Sparkles className="h-3 w-3" /> mitigation
          </div>
          <p className="text-[11px] text-slate-200 leading-relaxed">
            Class-balanced sample weights with an additional <span className="font-mono text-pink-200">1.7×</span> boost on{" "}
            <span className="font-mono text-emerald-200">top_performer</span> during training.
          </p>
        </div>
      </div>
    </div>
  );
}

function FindingCard({ title, body }: { title: string; body: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.5 }}
      className="card-surface p-5 relative overflow-hidden"
    >
      <div className="absolute -top-12 -right-8 h-32 w-32 rounded-full bg-pink-500/15 blur-2xl pointer-events-none" />
      <div className="relative">
        <Sparkles className="h-4 w-4 text-pink-300 mb-2" />
        <h3 className="font-display font-bold text-sm">{title}</h3>
        <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">{body}</p>
      </div>
    </motion.div>
  );
}

/* ============================================================
   Chapter 8 — Caveats
   ============================================================ */
function Chapter9Caveats() {
  return (
    <Chapter
      number="08" icon={AlertTriangle} accent="amber"
      label="honest caveats"
      title={<>What we'd <span className="text-amber-300">tell our future selves.</span></>}
      subtitle="Every result here has a confidence interval and an asterisk. Here are the asterisks."
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <CaveatCard
          title="Synthetic data"
          body="The Smadex dataset is synthetic. A predict-by-vertical-prior baseline gets ~0.30 macro-F1 for free. Our +0.35 lift comes mostly from early-life signal — real-world numbers may shift."
        />
        <CaveatCard
          title="Wide CI on top_performer"
          body="Top-performer F1 has a 95% bootstrap CI of [0.31, 0.89] (only n=11 in test). The 0.60 point estimate is directional, not gospel."
        />
        <CaveatCard
          title="Pause/Pivot is a Watch queue"
          body="Pause/Pivot recommendation precision ≈ 0.54. Don't run it autonomously — surface it for a human reviewer to act on."
        />
        <CaveatCard
          title="Cold-start gap"
          body="The 7-day early-life signal is the strongest feature. For brand-new creatives with zero impressions, the model degrades to the visual-rubric + metadata baseline (≈ 0.45 macro-F1)."
        />
      </div>

      <div className="mt-6 text-center">
        <p className="text-xs text-slate-500 uppercase tracking-[0.3em]">end of story</p>
      </div>
    </Chapter>
  );
}

function CaveatCard({ title, body }: { title: string; body: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.5 }}
      className="card-surface p-5 border-l-2 border-amber-500/40"
    >
      <h3 className="font-display font-bold text-sm flex items-center gap-1.5 text-amber-200">
        <AlertTriangle className="h-3.5 w-3.5" /> {title}
      </h3>
      <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">{body}</p>
    </motion.div>
  );
}
