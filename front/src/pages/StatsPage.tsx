import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  loadMetrics, loadEvalReport, loadPredictions,
  type FinalMetrics, type EvalReport, type Prediction, STATUS_COLORS, VERTICAL_COLORS,
} from "../lib/data";
import { Trophy, Activity, BadgeCheck, Layers } from "lucide-react";

const CLASS_NAMES = ["fatigued", "stable", "top_performer", "underperformer"] as const;

function MetricCard({
  label, value, sub, icon: Icon, accent,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: any;
  accent: string;
}) {
  return (
    <div className="card-surface relative overflow-hidden p-5">
      <div className={`absolute -top-16 -right-12 h-40 w-40 rounded-full ${accent} blur-3xl pointer-events-none`} />
      <div className="relative">
        <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
          <Icon className="h-3.5 w-3.5" /> {label}
        </div>
        <div className="mt-2 text-3xl font-extrabold tracking-tight">{value}</div>
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
                    className="p-2 text-center text-sm font-semibold rounded"
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

function StatusBar({ counts, total }: { counts: Record<string, number>; total: number }) {
  return (
    <div className="space-y-2.5">
      {CLASS_NAMES.map((cls) => {
        const n = counts[cls] || 0;
        const pct = total > 0 ? (n / total) * 100 : 0;
        const color = STATUS_COLORS[cls];
        return (
          <div key={cls}>
            <div className="flex justify-between text-xs mb-1">
              <span className={`${color.fg} capitalize font-medium`}>{cls.replace("_", " ")}</span>
              <span className="text-slate-400">{n} · {pct.toFixed(1)}%</span>
            </div>
            <div className="h-2 rounded-full bg-white/5 overflow-hidden">
              <div
                className={`h-full ${color.bg.replace("/15", "/60")}`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function VerticalBars({ predictions }: { predictions: Prediction[] }) {
  const test = predictions.filter((p) => p.split === "test");
  const verticals = Array.from(new Set(test.map((p) => p.vertical))).sort();
  const f1ByVertical = verticals.map((v) => {
    const sub = test.filter((p) => p.vertical === v);
    const correct = sub.filter((p) => p.pred_status === p.true_status).length;
    return { vertical: v, n: sub.length, accuracy: sub.length ? correct / sub.length : 0 };
  }).sort((a, b) => b.accuracy - a.accuracy);

  return (
    <div className="space-y-2">
      {f1ByVertical.map((v) => (
        <div key={v.vertical}>
          <div className="flex justify-between text-xs mb-1">
            <span
              className="capitalize font-medium"
              style={{ color: VERTICAL_COLORS[v.vertical] || "#cbd5e1" }}
            >
              {v.vertical.replace("_", " ")}
            </span>
            <span className="text-slate-400">acc {(v.accuracy * 100).toFixed(0)}% · n={v.n}</span>
          </div>
          <div className="h-2 rounded-full bg-white/5 overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${v.accuracy * 100}%`,
                backgroundColor: VERTICAL_COLORS[v.vertical] || "#94a3b8",
                opacity: 0.75,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

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
    return (
      <main className="pt-28 container-narrow"><p className="text-slate-400">loading…</p></main>
    );
  }

  // Compute test class counts
  const test = preds.filter((p) => p.split === "test");
  const counts = test.reduce<Record<string, number>>((acc, p) => {
    acc[p.true_status] = (acc[p.true_status] || 0) + 1; return acc;
  }, {});

  return (
    <main className="pt-28 pb-16">
      <div className="container-narrow">
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        >
          <span className="stat-pill">
            <Activity className="h-3 w-3" /> held-out test set · n=216
          </span>
          <h1 className="mt-3 text-3xl sm:text-4xl font-bold tracking-tight">Model performance</h1>
          <p className="mt-2 text-slate-400 text-sm max-w-xl">
            Numbers below are from the production model (refit on train ∪ val, n=860).
            Test set was touched once. No temperature scaling — raw ensemble probabilities.
          </p>
        </motion.div>

        {/* Headline metrics */}
        <div className="mt-10 grid grid-cols-2 lg:grid-cols-4 gap-4">
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
            label="Health Score Spearman" value={evalReport.health_score_spearman.toFixed(2)}
            sub={`fatigue 4-bucket F1 ${evalReport.fatigue_4bucket_test_f1.toFixed(2)}`}
            icon={Activity} accent="bg-pink-500/30"
          />
        </div>

        {/* Per-class confusion + class distribution */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="card-surface p-6">
            <h3 className="text-base font-semibold mb-3">Confusion matrix (test)</h3>
            <ConfusionMatrix
              matrix={metrics.test.confusion_matrix}
              classes={metrics.test.class_names}
            />
            <p className="mt-3 text-xs text-slate-400">
              Diagonal = correct predictions. Off-diagonal red intensity = confusion magnitude.
            </p>
          </div>

          <div className="card-surface p-6">
            <h3 className="text-base font-semibold mb-3">Test class distribution</h3>
            <StatusBar counts={counts} total={test.length} />
            <p className="mt-3 text-xs text-slate-400">
              Severely imbalanced (16:1 ratio); model uses class-balanced sample weights with
              an additional 1.7× boost on top_performer.
            </p>
          </div>
        </div>

        {/* Per-vertical + per-model */}
        <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="card-surface p-6">
            <h3 className="text-base font-semibold mb-3">Per-vertical accuracy (test)</h3>
            <VerticalBars predictions={preds} />
          </div>

          <div className="card-surface p-6">
            <h3 className="text-base font-semibold mb-3">Per-model val macro-F1</h3>
            <div className="space-y-2.5">
              {Object.keys(metrics.val_macro_f1_per_model).length === 0 ? (
                <p className="text-xs text-slate-400 italic">
                  Per-model val scores omitted in --final mode (val is part of training).
                  See clean_metrics.json for val-tuned per-model breakdown.
                </p>
              ) : (
                Object.entries(metrics.val_macro_f1_per_model)
                  .sort(([, a], [, b]) => b - a)
                  .map(([name, f1]) => (
                    <div key={name}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-medium text-slate-200">{name}</span>
                        <span className="text-slate-400">{f1.toFixed(4)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-brand-500 to-pink-500"
                          style={{ width: `${f1 * 100}%` }}
                        />
                      </div>
                    </div>
                  ))
              )}
            </div>
          </div>
        </div>

        {/* Honest caveats footer */}
        <div className="mt-6 card-surface p-6 text-sm text-slate-300">
          <h3 className="text-base font-semibold mb-2">Honest caveats</h3>
          <ul className="space-y-1.5 list-disc list-inside text-slate-400">
            <li>Top-performer F1 has 95% CI [0.31, 0.89] (n=11 in test) — point estimate is directional.</li>
            <li>Pause/Pivot recommendation precision ≈ 0.54; should run as a Watch queue, not autonomous action.</li>
            <li>Dataset is synthetic; vertical→status confound (χ² ≈ 335) means a vertical-prior baseline gets ~0.30 macro-F1 for free.</li>
          </ul>
        </div>
      </div>
    </main>
  );
}
