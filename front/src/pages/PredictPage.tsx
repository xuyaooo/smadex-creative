import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  loadPredictions, loadMetadata, type Prediction, type Metadata,
  STATUS_COLORS, actionFromHealth, actionColor,
} from "../lib/data";
import { Zap, Target, ListChecks, ArrowDownRight } from "lucide-react";

/**
 * Prediction page — picks the closest precomputed prediction by feature
 * similarity, since we're running fully static (no backend).
 *
 * For a hackathon demo this is "good enough": with 1,076 reference points
 * across all 6 verticals × 4 formats × 7 colors × 4 hook types, any input
 * combination has a close neighbor.
 */
export default function PredictPage() {
  const [preds, setPreds] = useState<Prediction[] | null>(null);
  const [meta, setMeta] = useState<Metadata | null>(null);

  // Form state
  const [vertical, setVertical] = useState("gaming");
  const [format, setFormat] = useState("rewarded_video");
  const [color, setColor] = useState("purple");
  const [hookType, setHookType] = useState("");
  const [theme, setTheme] = useState("");
  const [emotionalTone, setEmotionalTone] = useState("");
  const [hasPrice, setHasPrice] = useState(false);
  const [hasDiscount, setHasDiscount] = useState(false);
  const [hasGameplay, setHasGameplay] = useState(true);
  const [hasUGC, setHasUGC] = useState(false);
  const [earlyCtr, setEarlyCtr] = useState(0.012);
  const [earlyImp, setEarlyImp] = useState(200000);
  const [earlySpend, setEarlySpend] = useState(2000);
  const [earlyRev, setEarlyRev] = useState(8000);
  const [duration, setDuration] = useState(15);

  useEffect(() => {
    Promise.all([loadPredictions(), loadMetadata()]).then(([p, m]) => {
      setPreds(p); setMeta(m);
      // Default-fill the 1st options
      if (m.hook_types[0]) setHookType(m.hook_types[0]);
      if (m.themes[0])     setTheme(m.themes[0]);
      if (m.emotional_tones[0]) setEmotionalTone(m.emotional_tones[0]);
    });
  }, []);

  // Find the nearest neighbor in the predictions corpus.
  // Distance: weighted equality of categoricals + log-scaled numerics.
  const neighbor = useMemo<Prediction | null>(() => {
    if (!preds) return null;
    const safeLog = (x: number) => Math.log(Math.max(x, 1e-6));
    const num = (a: number, b: number) => Math.abs(safeLog(a + 1) - safeLog(b + 1));
    let best: Prediction | null = null;
    let bestD = Infinity;
    for (const p of preds) {
      let d = 0;
      d += p.vertical === vertical ? 0 : 3;
      d += p.format === format ? 0 : 2;
      d += p.dominant_color === color ? 0 : 1;
      d += p.hook_type === hookType ? 0 : 0.7;
      d += p.theme === theme ? 0 : 0.5;
      d += p.emotional_tone === emotionalTone ? 0 : 0.5;
      d += +(p.has_price !== (hasPrice ? 1 : 0));
      d += +(p.has_discount_badge !== (hasDiscount ? 1 : 0));
      d += +(p.has_gameplay !== (hasGameplay ? 1 : 0));
      d += +(p.has_ugc_style !== (hasUGC ? 1 : 0));
      d += num(p.early_ctr, earlyCtr) * 0.8;
      d += num(p.early_imp, earlyImp) * 0.4;
      d += num(p.early_spend, earlySpend) * 0.3;
      d += num(p.early_revenue, earlyRev) * 0.3;
      d += num(p.duration_sec, duration) * 0.2;
      if (d < bestD) { bestD = d; best = p; }
    }
    return best;
  }, [
    preds, vertical, format, color, hookType, theme, emotionalTone,
    hasPrice, hasDiscount, hasGameplay, hasUGC,
    earlyCtr, earlyImp, earlySpend, earlyRev, duration,
  ]);

  if (!preds || !meta) {
    return (
      <main className="pt-28 container-narrow"><p className="text-slate-400">loading…</p></main>
    );
  }

  const action = neighbor ? actionFromHealth(neighbor.health_score) : "Watch";
  const ac = actionColor(action);

  // Build counterfactual recommendations: try alternate value for each
  // mutable feature and report the one giving the highest health.
  const altSuggestion = useMemo(() => {
    if (!preds || !neighbor) return null;
    const baseScore = neighbor.health_score;
    const tries: { feat: string; from: string; to: string; score: number }[] = [];
    const dims: Array<["vertical" | "format" | "dominant_color" | "hook_type" | "theme" | "emotional_tone", string]> = [
      ["vertical", vertical], ["format", format], ["dominant_color", color],
      ["hook_type", hookType], ["theme", theme], ["emotional_tone", emotionalTone],
    ];
    for (const [feat, current] of dims) {
      const opts = (
        feat === "vertical" ? meta.verticals :
        feat === "format" ? meta.formats :
        feat === "dominant_color" ? meta.dominant_colors :
        feat === "hook_type" ? meta.hook_types :
        feat === "theme" ? meta.themes :
        meta.emotional_tones
      );
      for (const alt of opts) {
        if (alt === current) continue;
        // Find nearest neighbor when this feature is altered
        let best: Prediction | null = null; let bestD = Infinity;
        for (const p of preds) {
          if (p[feat] !== alt) continue;
          let d = 0;
          d += +(p.vertical !== vertical) * 3;
          d += +(p.format !== format) * 2;
          d += +(p.dominant_color !== color) * 1;
          d += +(p.hook_type !== hookType) * 0.7;
          d += +(p.theme !== theme) * 0.5;
          d += +(p.emotional_tone !== emotionalTone) * 0.5;
          // override the changed dim
          d -= +(p[feat] !== current) * (feat === "vertical" ? 3 : feat === "format" ? 2 : 1);
          if (d < bestD) { bestD = d; best = p; }
        }
        if (best && best.health_score > baseScore) {
          tries.push({ feat, from: current, to: alt, score: best.health_score });
        }
      }
    }
    return tries.sort((a, b) => b.score - a.score).slice(0, 3);
  }, [preds, meta, neighbor, vertical, format, color, hookType, theme, emotionalTone]);

  return (
    <main className="pt-28 pb-16">
      <div className="container-narrow">
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        >
          <span className="stat-pill"><Zap className="h-3 w-3" /> creative scoring</span>
          <h1 className="mt-3 text-3xl sm:text-4xl font-bold tracking-tight">Score a creative</h1>
          <p className="mt-2 text-slate-400 text-sm max-w-xl">
            Configure a creative on the left → the model returns a Health Score, predicted status,
            recommended action, and a "what to change" suggestion. Runs fully client-side
            using precomputed predictions.
          </p>
        </motion.div>

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* FORM */}
          <div className="card-surface p-6 space-y-5">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Vertical">
                <Select value={vertical} onChange={setVertical} options={meta.verticals} />
              </Field>
              <Field label="Format">
                <Select value={format} onChange={setFormat} options={meta.formats} />
              </Field>
              <Field label="Dominant color">
                <Select value={color} onChange={setColor} options={meta.dominant_colors} />
              </Field>
              <Field label="Hook type">
                <Select value={hookType} onChange={setHookType} options={meta.hook_types} />
              </Field>
              <Field label="Theme">
                <Select value={theme} onChange={setTheme} options={meta.themes} />
              </Field>
              <Field label="Emotional tone">
                <Select value={emotionalTone} onChange={setEmotionalTone} options={meta.emotional_tones} />
              </Field>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Toggle label="Shows price" value={hasPrice} onChange={setHasPrice} />
              <Toggle label="Discount badge" value={hasDiscount} onChange={setHasDiscount} />
              <Toggle label="Gameplay footage" value={hasGameplay} onChange={setHasGameplay} />
              <Toggle label="UGC style" value={hasUGC} onChange={setHasUGC} />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Field label={`Early CTR — ${(earlyCtr * 100).toFixed(2)}%`}>
                <Range min={0} max={0.04} step={0.001} value={earlyCtr} onChange={setEarlyCtr} />
              </Field>
              <Field label={`Duration ${duration}s`}>
                <Range min={5} max={45} step={1} value={duration} onChange={setDuration} />
              </Field>
              <Field label={`Early imps ${earlyImp.toLocaleString()}`}>
                <Range min={10000} max={2000000} step={10000} value={earlyImp} onChange={setEarlyImp} />
              </Field>
              <Field label={`Early spend $${earlySpend.toLocaleString()}`}>
                <Range min={100} max={50000} step={100} value={earlySpend} onChange={setEarlySpend} />
              </Field>
              <Field label={`Early revenue $${earlyRev.toLocaleString()}`} span={2}>
                <Range min={0} max={200000} step={500} value={earlyRev} onChange={setEarlyRev} />
              </Field>
            </div>
          </div>

          {/* OUTPUT */}
          <div className="space-y-4">
            {/* Health score */}
            <div className="card-surface p-6 relative overflow-hidden">
              <div className="absolute -top-32 -right-20 h-72 w-72 rounded-full bg-brand-500/30 blur-3xl pointer-events-none" />
              <div className="relative">
                <div className="text-xs uppercase tracking-wider text-slate-400">Creative Health Score</div>
                <div className="mt-2 flex items-end gap-3">
                  <div className="text-6xl font-extrabold tracking-tight">{neighbor?.health_score.toFixed(0) ?? "—"}</div>
                  <div className="text-2xl text-slate-400 mb-2">/100</div>
                </div>
                <div className="mt-3 h-2 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-brand-500 to-pink-500"
                    style={{ width: `${neighbor?.health_score ?? 0}%` }}
                  />
                </div>
                <div className="mt-3 flex items-center gap-2">
                  <span className={`text-sm font-bold px-3 py-1 rounded-full ${ac.fg} ${ac.bg}`}>
                    <Target className="inline h-3.5 w-3.5 mr-1" />
                    {action}
                  </span>
                  <span className="text-xs text-slate-400">recommended action</span>
                </div>
              </div>
            </div>

            {/* Predicted status + probabilities */}
            <div className="card-surface p-6">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
                <ListChecks className="h-3.5 w-3.5" /> predicted status
              </div>
              <div className="mt-2 text-2xl font-bold capitalize">{(neighbor?.pred_status ?? "—").replace("_", " ")}</div>
              <div className="mt-4 space-y-2">
                {neighbor &&
                  ([
                    ["top_performer", neighbor.p_top],
                    ["stable",        neighbor.p_stable],
                    ["fatigued",      neighbor.p_fatigued],
                    ["underperformer",neighbor.p_under],
                  ] as [keyof typeof STATUS_COLORS, number][]).map(([n, v]) => (
                    <div key={n}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className={`${STATUS_COLORS[n].fg} capitalize`}>{n.replace("_", " ")}</span>
                        <span className="text-slate-400 tabular-nums">{(v * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-white/5">
                        <div
                          className={`h-full rounded-full ${STATUS_COLORS[n].bg.replace("/15", "/70")}`}
                          style={{ width: `${v * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Counterfactual */}
            {altSuggestion && altSuggestion.length > 0 && (
              <div className="card-surface p-6">
                <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
                  <ArrowDownRight className="h-3.5 w-3.5" /> what to change
                </div>
                <p className="mt-2 text-xs text-slate-400">
                  Smallest single-feature changes that would lift the score:
                </p>
                <div className="mt-3 space-y-2">
                  {altSuggestion.map((s, i) => (
                    <div key={i} className="rounded-lg bg-white/5 px-3 py-2 text-sm">
                      <span className="text-slate-400 text-xs">change </span>
                      <span className="font-semibold">{s.feat.replace("_", " ")}</span>
                      <span className="text-slate-400 text-xs"> from </span>
                      <span className="text-rose-300">{s.from}</span>
                      <span className="text-slate-400 text-xs"> → </span>
                      <span className="text-emerald-300">{s.to}</span>
                      <span className="ml-2 text-xs text-slate-400">→ score ≈ {s.score.toFixed(0)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <p className="mt-8 text-xs text-slate-500 text-center">
          Inference runs client-side via nearest-neighbor lookup over 1,076 precomputed predictions.
          For exact model inference on novel inputs, point the app at the FastAPI backend at
          <code className="mx-1 px-1.5 py-0.5 rounded bg-white/5">localhost:8000</code>.
        </p>
      </div>
    </main>
  );
}

function Field({ label, span = 1, children }: { label: string; span?: 1 | 2; children: React.ReactNode }) {
  return (
    <label className={`block ${span === 2 ? "col-span-2" : ""}`}>
      <span className="text-[11px] uppercase tracking-wider text-slate-400">{label}</span>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/40 capitalize"
    >
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm transition
                  ${value ? "bg-brand-500/15 border-brand-500/40 text-brand-200" : "bg-white/5 border-white/10 text-slate-300"}`}
    >
      <span>{label}</span>
      <span className={`h-4 w-7 rounded-full p-0.5 transition ${value ? "bg-brand-500" : "bg-white/15"}`}>
        <span className={`block h-3 w-3 rounded-full bg-white transition ${value ? "translate-x-3" : "translate-x-0"}`} />
      </span>
    </button>
  );
}

function Range({ value, onChange, min, max, step }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step: number;
}) {
  return (
    <input
      type="range" min={min} max={max} step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full accent-brand-500"
    />
  );
}
