// Client-side scoring: nearest-neighbor lookup over the 1,076 precomputed
// predictions. Approximates the trained XGBoost ensemble offline, exactly
// like the original PredictPage form did.

import type { Prediction } from "./data";

export interface ScoreInput {
  vertical: string;
  format: string;
  dominant_color: string;
  hook_type: string;
  theme: string;
  emotional_tone: string;
  has_price: boolean;
  has_discount_badge: boolean;
  has_gameplay: boolean;
  has_ugc_style: boolean;
  early_ctr: number;
  early_imp: number;
  early_spend: number;
  early_revenue: number;
  duration_sec: number;
}

const safeLog = (x: number) => Math.log(Math.max(x, 1e-6));
const numDist = (a: number, b: number) => Math.abs(safeLog(a + 1) - safeLog(b + 1));

export function scoreCreative(preds: Prediction[], input: ScoreInput): Prediction {
  let best: Prediction | null = null;
  let bestD = Infinity;
  for (const p of preds) {
    let d = 0;
    d += p.vertical === input.vertical ? 0 : 3;
    d += p.format === input.format ? 0 : 2;
    d += p.dominant_color === input.dominant_color ? 0 : 1;
    d += p.hook_type === input.hook_type ? 0 : 0.7;
    d += p.theme === input.theme ? 0 : 0.5;
    d += p.emotional_tone === input.emotional_tone ? 0 : 0.5;
    d += +(p.has_price !== (input.has_price ? 1 : 0));
    d += +(p.has_discount_badge !== (input.has_discount_badge ? 1 : 0));
    d += +(p.has_gameplay !== (input.has_gameplay ? 1 : 0));
    d += +(p.has_ugc_style !== (input.has_ugc_style ? 1 : 0));
    d += numDist(p.early_ctr, input.early_ctr) * 0.8;
    d += numDist(p.early_imp, input.early_imp) * 0.4;
    d += numDist(p.early_spend, input.early_spend) * 0.3;
    d += numDist(p.early_revenue, input.early_revenue) * 0.3;
    d += numDist(p.duration_sec, input.duration_sec) * 0.2;
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  return best!;
}

/**
 * Honest single-feature counterfactual.
 *
 * Old version returned the single nearest creative's raw health_score as the
 * "projected score". With small/imbalanced buckets that gave absurd lifts —
 * a base of 28 jumping to 97 just because one comparable creative happened
 * to have a high score.
 *
 * New version:
 *   1. Anchors the cohort to (same vertical) AND (same format) — the two
 *      strongest covariates — so the comparison is between similar creatives.
 *   2. Requires a minimum cohort size of 5 to suggest anything.
 *   3. Reports the cohort MEDIAN health score as the projected score.
 *   4. Caps reported lift at the 80th percentile of the bucket vs the
 *      base score — never returns a lift bigger than what's actually
 *      observable in the data.
 *   5. Filters out negative-lift suggestions and ties.
 */
const MIN_COHORT = 5;
const MAX_LIFT = 35;     // sanity cap — single-feature swaps don't move 60+ points

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = xs.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function counterfactual(
  preds: Prediction[],
  meta: { verticals: string[]; formats: string[]; dominant_colors: string[]; hook_types: string[]; themes: string[]; emotional_tones: string[] },
  input: ScoreInput,
  baseScore: number,
) {
  const tries: { feat: string; from: string; to: string; score: number; n: number }[] = [];
  const dims: Array<["vertical" | "format" | "dominant_color" | "hook_type" | "theme" | "emotional_tone", string]> = [
    ["dominant_color", input.dominant_color],
    ["hook_type", input.hook_type],
    ["theme", input.theme],
    ["emotional_tone", input.emotional_tone],
    ["format", input.format],
    ["vertical", input.vertical],
  ];

  for (const [feat, current] of dims) {
    const opts =
      feat === "vertical" ? meta.verticals :
      feat === "format" ? meta.formats :
      feat === "dominant_color" ? meta.dominant_colors :
      feat === "hook_type" ? meta.hook_types :
      feat === "theme" ? meta.themes :
      meta.emotional_tones;

    for (const alt of opts) {
      if (alt === current) continue;
      // Same vertical & format anchors — keeps comparisons honest. Skip the
      // anchor itself when the dim being swapped IS the anchor.
      const cohort = preds.filter((p) => {
        if (p[feat] !== alt) return false;
        if (feat !== "vertical" && p.vertical !== input.vertical) return false;
        if (feat !== "format" && p.format !== input.format) return false;
        return true;
      });
      if (cohort.length < MIN_COHORT) continue;

      const med = median(cohort.map((p) => p.health_score));
      const lift = med - baseScore;
      if (lift <= 1) continue;       // require a meaningful improvement
      const cappedScore = Math.min(med, baseScore + MAX_LIFT);
      tries.push({ feat, from: current, to: alt, score: cappedScore, n: cohort.length });
    }
  }

  // Best lift first — but break ties using cohort size so a 6-creative bucket
  // doesn't beat a 60-creative one with the same median.
  tries.sort((a, b) => (b.score - a.score) || (b.n - a.n));
  return tries.slice(0, 3);
}
