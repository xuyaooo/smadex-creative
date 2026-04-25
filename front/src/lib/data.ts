// Lightweight data loader — fetches the precomputed bundle from /public/data/*.json

export type ClassName = "fatigued" | "stable" | "top_performer" | "underperformer";

export interface Prediction {
  creative_id: number;
  campaign_id: number;
  split: "train" | "val" | "test";
  vertical: string;
  format: string;
  theme: string;
  hook_type: string;
  dominant_color: string;
  emotional_tone: string;
  has_price: 0 | 1;
  has_discount_badge: 0 | 1;
  has_gameplay: 0 | 1;
  has_ugc_style: 0 | 1;
  duration_sec: number;
  early_imp: number;
  early_clicks: number;
  early_ctr: number;
  early_revenue: number;
  early_spend: number;
  true_status: ClassName;
  pred_status: ClassName;
  p_top: number;
  p_stable: number;
  p_fatigued: number;
  p_under: number;
  health_score: number;
  true_fatigue: "never" | "early" | "standard" | "late";
  pred_fatigue: "never" | "early" | "standard" | "late";
  cluster: number;
}

export interface Metadata {
  class_names: string[];
  verticals: string[];
  formats: string[];
  themes: string[];
  hook_types: string[];
  dominant_colors: string[];
  emotional_tones: string[];
  feats: string[];
  cat_cols: string[];
  num_cols: string[];
  fatigue_classes: string[];
}

export interface FinalMetrics {
  data: { train: number; val: number; test: number; n_features: number };
  timings_seconds: Record<string, number>;
  val_macro_f1_per_model: Record<string, number>;
  temperature: number;
  test: {
    macro_f1: number;
    weighted_f1: number;
    accuracy: number;
    log_loss: number;
    ece: number;
    confusion_matrix: number[][];
    class_names: string[];
  };
  fatigue_4bucket: { val_macro_f1: number; test_macro_f1: number };
}

export interface EvalReport {
  data: { train: number; val: number; test: number; n_features: number };
  training_seconds_total: number;
  temperature_scaling: string;
  test: {
    macro_f1: number;
    weighted_f1: number;
    accuracy: number;
    log_loss: number;
    ece: number;
    auc_top_performer: number;
    auc_underperformer: number;
  };
  fatigue_4bucket_test_f1: number;
  health_score_spearman: number;
  best_per_vertical: { vertical: string; n: number; "macro-F1": number; accuracy: number };
  worst_per_vertical: { vertical: string; n: number; "macro-F1": number; accuracy: number };
  artifacts_dir: string;
}

let cachedPredictions: Prediction[] | null = null;
let cachedMeta: Metadata | null = null;
let cachedMetrics: FinalMetrics | null = null;
let cachedEval: EvalReport | null = null;

const BASE = (import.meta as any).env?.BASE_URL ?? "/";
const url = (p: string) => `${BASE}data/${p}`.replace(/\/\//g, "/");

export async function loadPredictions(): Promise<Prediction[]> {
  if (cachedPredictions) return cachedPredictions;
  const res = await fetch(url("predictions.json"));
  cachedPredictions = (await res.json()) as Prediction[];
  return cachedPredictions;
}

export async function loadMetadata(): Promise<Metadata> {
  if (cachedMeta) return cachedMeta;
  const res = await fetch(url("metadata.json"));
  cachedMeta = (await res.json()) as Metadata;
  return cachedMeta;
}

export async function loadMetrics(): Promise<FinalMetrics> {
  if (cachedMetrics) return cachedMetrics;
  const res = await fetch(url("final_metrics.json"));
  cachedMetrics = (await res.json()) as FinalMetrics;
  return cachedMetrics;
}

export async function loadEvalReport(): Promise<EvalReport> {
  if (cachedEval) return cachedEval;
  const res = await fetch(url("eval_report.json"));
  cachedEval = (await res.json()) as EvalReport;
  return cachedEval;
}

// ---------- Helpers ----------

export const STATUS_COLORS: Record<ClassName, { fg: string; bg: string; ring: string }> = {
  top_performer:  { fg: "text-emerald-300", bg: "bg-emerald-500/15", ring: "ring-emerald-500/30" },
  stable:         { fg: "text-sky-300",     bg: "bg-sky-500/15",     ring: "ring-sky-500/30" },
  fatigued:       { fg: "text-amber-300",   bg: "bg-amber-500/15",   ring: "ring-amber-500/30" },
  underperformer: { fg: "text-rose-300",    bg: "bg-rose-500/15",    ring: "ring-rose-500/30" },
};

export const VERTICAL_COLORS: Record<string, string> = {
  gaming: "#a78bfa", travel: "#22d3ee", fintech: "#34d399",
  ecommerce: "#fbbf24", food_delivery: "#f87171", entertainment: "#f472b6",
};

export function actionFromHealth(score: number): "Scale" | "Maintain" | "Watch" | "Pause/Pivot" {
  if (score >= 75) return "Scale";
  if (score >= 50) return "Maintain";
  if (score >= 25) return "Watch";
  return "Pause/Pivot";
}

export function actionColor(action: string): { fg: string; bg: string } {
  switch (action) {
    case "Scale":       return { fg: "text-emerald-200", bg: "bg-emerald-500/20" };
    case "Maintain":    return { fg: "text-sky-200",     bg: "bg-sky-500/20" };
    case "Watch":       return { fg: "text-amber-200",   bg: "bg-amber-500/20" };
    case "Pause/Pivot": return { fg: "text-rose-200",    bg: "bg-rose-500/20" };
    default:            return { fg: "text-slate-200",   bg: "bg-white/10" };
  }
}
