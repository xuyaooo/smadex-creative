"""Benchmark candidate (model × prompt) combos on a 12-creative ground-truth set.

Quality = correlation of rubric scores with synthetic ground-truth metadata that
we know is in the dataset (faces_count, has_discount_badge, text_density, etc.).
A good model+prompt combo should:
  1. Use the full 0-10 range (per-dim std > 1.5 across the 12 creatives).
  2. Recover metadata correlations: e.g.,
       - faces_count          ~ face_count_visual         (Spearman > 0.7)
       - has_discount_badge   ~ urgency_signal             (>0.6)
       - text_density         ~ text_density_visual        (>0.5)
       - clutter_score (high) ~ composition_balance (low)  (<-0.3)
       - novelty_score        ~ novelty_visual             (>0.4)
       - brand_visibility_sc  ~ brand_visibility / 10      (>0.4)

Outputs a leaderboard so you can pick the winning combo for the full run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.training.openrouter_rubric import OpenRouterRubric, RUBRIC_NAMES

BENCH_IDS = [500000, 500011, 500016, 500200, 500356, 500400,
             500500, 500700, 500725, 500800, 500903, 500995]

# (rubric_dim, ground_truth_col, sign, target_min_corr, weight)
GROUND_TRUTH_PAIRS = [
    ("face_count_visual",    "faces_count",            +1, 0.7, 2.0),
    ("urgency_signal",       "has_discount_badge",     +1, 0.5, 1.5),
    ("urgency_signal",       "has_price",              +1, 0.3, 1.0),
    ("text_density_visual",  "text_density",           +1, 0.4, 1.5),
    ("composition_balance",  "clutter_score",          -1, 0.3, 1.0),
    ("novelty_visual",       "novelty_score",          +1, 0.3, 1.0),
    ("brand_visibility",     "brand_visibility_score", +1, 0.3, 1.0),
]


def score_one_combo(api_key: str, model: str, asset_dir: Path, workers: int = 12):
    """Score the 12 benchmark creatives. Returns (cid → 15-dim dict)."""
    rubric = OpenRouterRubric(model=model, api_key=api_key, requests_per_minute=120)
    out: dict = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(rubric.score_one, asset_dir / f"creative_{cid}.png"): cid
                   for cid in BENCH_IDS}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                scores = fut.result()
                if scores is not None:
                    out[cid] = scores
            except Exception as e:
                print(f"  {model} cid={cid}: {e}")
    return out


def evaluate(scores_by_cid: dict, gt_df: pd.DataFrame) -> dict:
    if not scores_by_cid:
        return {"score": 0.0, "n_calls_ok": 0, "details": {}}

    score_df = pd.DataFrame.from_dict(scores_by_cid, orient="index").reindex(BENCH_IDS).dropna()
    if len(score_df) < 8:
        return {"score": 0.0, "n_calls_ok": len(score_df), "details": {}}

    # Per-dim std → measures range usage
    avg_std = float(score_df.std().mean())
    range_score = min(1.0, avg_std / 2.0)

    # Correlations
    aligned = score_df.merge(gt_df, left_index=True, right_index=True, how="inner")
    corr_terms = []
    details = {}
    for rubric_col, gt_col, sign, target, weight in GROUND_TRUTH_PAIRS:
        if rubric_col not in aligned.columns or gt_col not in aligned.columns:
            continue
        try:
            rho, _ = spearmanr(aligned[rubric_col], aligned[gt_col])
            if np.isnan(rho):
                rho = 0.0
        except Exception:
            rho = 0.0
        signed = rho * sign
        corr_terms.append((signed, weight))
        details[f"{rubric_col} vs {gt_col}"] = round(signed, 3)

    if corr_terms:
        weighted = sum(s * w for s, w in corr_terms) / sum(w for _, w in corr_terms)
    else:
        weighted = 0.0

    # Final composite
    composite = 0.4 * range_score + 0.6 * max(0.0, weighted)
    return {
        "score": round(composite, 3),
        "n_calls_ok": len(score_df),
        "avg_std": round(avg_std, 2),
        "weighted_corr": round(weighted, 3),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=[
        "google/gemini-2.0-flash-001",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
        "anthropic/claude-haiku-4.5",
    ])
    args = parser.parse_args()

    import yaml, os
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY or pass --api-key.")
    asset_dir = (Path(args.config).parent / cfg["data"]["assets_dir"]).resolve()

    creatives = pd.read_csv("../creatives.csv").set_index("creative_id")
    gt_df = creatives.loc[BENCH_IDS, [
        "faces_count", "has_discount_badge", "has_price",
        "text_density", "clutter_score", "novelty_score", "brand_visibility_score",
    ]]

    leaderboard = []
    for model in args.models:
        print(f"\n=== {model} ===")
        t0 = time.time()
        try:
            scores = score_one_combo(api_key, model, asset_dir, workers=12)
        except Exception as e:
            print(f"  FAILED to init: {e}")
            leaderboard.append({"model": model, "score": 0.0, "error": str(e)})
            continue
        elapsed = time.time() - t0
        result = evaluate(scores, gt_df)
        result["model"] = model
        result["seconds"] = round(elapsed, 1)
        print(f"  ok={result['n_calls_ok']}/12  std={result.get('avg_std','?')}  "
              f"corr={result.get('weighted_corr','?')}  composite={result['score']}  "
              f"time={result['seconds']}s")
        for k, v in result.get("details", {}).items():
            print(f"    {k}: {v}")
        leaderboard.append(result)

    print("\n" + "=" * 60)
    print("LEADERBOARD (composite higher is better)")
    print("=" * 60)
    leaderboard.sort(key=lambda r: r.get("score", 0), reverse=True)
    for r in leaderboard:
        print(f"  {r.get('score',0):.3f}  {r['model']}  "
              f"(corr={r.get('weighted_corr','?')}, std={r.get('avg_std','?')}, "
              f"ok={r.get('n_calls_ok','?')}, t={r.get('seconds','?')}s)")


if __name__ == "__main__":
    main()
