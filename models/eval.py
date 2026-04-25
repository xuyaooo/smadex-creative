"""
Full end-to-end evaluation: retrain, then test every model component.

Reports:
  1. Training metrics (OOF perf MAE, OOF status F1, calibration ECE)
  2. Per-operation latency (cold start, health_score, find_similar, cluster_info, explain)
  3. Action accuracy (Health Score → action, vs ground-truth status mapping)
  4. Recommender quality (mean status match in top-5 similar)
  5. Cluster quality (vertical purity, mean cluster size)
  6. Four sample explanations (one per status class)
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.inference.pipeline import CreativeIntelligencePipeline


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    p = CreativeIntelligencePipeline("config.yaml")
    t0 = time.time()
    p._ensure_models()
    cold_start = time.time() - t0

    section("1. PIPELINE READY")
    print(f"  Cold start:        {cold_start:.2f} s")
    print(f"  Master rows:       {len(p._master_df):,}")
    print(f"  Daily rows:        {len(p._daily_df):,}")
    print(f"  CLIP embeddings:   {len(p.embedding_cache.get_all()[1]):,}")
    print(f"  Early features:    {p._n_early_features} dims")
    print(f"  Rubric features:   {p._n_rubric_features} dims")
    print(f"  Clusters loaded:   {len(set(p._cluster_by_cid.values()))}")
    print(f"  Cluster names:     {len(p._cluster_names)}")
    print(f"  KNN verticals:     {len(p._knn_by_vertical) - 1}")
    print(f"  Temperature T:     {p._temperature.T:.3f}")

    # ---------- 2. Latency benchmark over a 30-creative sample ----------
    section("2. LATENCY (30-creative sample, mean ms per call)")
    sample_ids = p._master_df.sample(30, random_state=42)["creative_id"].astype(int).tolist()
    timings = {"health_score": [], "find_similar": [], "cluster_info": [], "explain": []}
    for cid in sample_ids:
        for op in timings:
            t0 = time.time()
            getattr(p, op)(cid)
            timings[op].append((time.time() - t0) * 1000)
    for op, vals in timings.items():
        v = np.array(vals)
        print(f"  {op:<14}  mean={v.mean():>5.1f}  p50={np.percentile(v,50):>5.1f}  "
              f"p95={np.percentile(v,95):>5.1f}  max={v.max():>5.1f}")

    # ---------- 3. Action accuracy ----------
    section("3. ACTION ACCURACY (Health Score → action vs true status)")
    # True status → expected action
    expected_action = {
        "top_performer":  "Scale",
        "stable":         "Continue",
        "fatigued":       "Pause",
        "underperformer": "Pivot",
    }
    relaxed_match = {
        # Continue/Scale acceptable for top_performer; Pivot/Pause for fatigued; etc.
        "top_performer":  {"Scale", "Continue"},
        "stable":         {"Continue", "Scale"},
        "fatigued":       {"Pause", "Pivot"},
        "underperformer": {"Pause", "Pivot"},
    }
    n_total = len(p._master_df)
    strict_correct = 0
    relaxed_correct = 0
    by_status: dict = {s: [0, 0] for s in expected_action}  # [strict, n]
    confusion = Counter()
    sample_for_speed = p._master_df.sample(min(200, n_total), random_state=42)

    t0 = time.time()
    for _, row in sample_for_speed.iterrows():
        cid = int(row["creative_id"])
        true = row["creative_status"]
        h = p.health_score(cid)
        pred = h["action"]
        confusion[(true, pred)] += 1
        if pred == expected_action[true]:
            strict_correct += 1
            by_status[true][0] += 1
        if pred in relaxed_match[true]:
            relaxed_correct += 1
        by_status[true][1] += 1
    elapsed = time.time() - t0
    n = len(sample_for_speed)

    print(f"  Sampled n={n}, total time {elapsed:.1f}s ({1000*elapsed/n:.0f} ms/call)")
    print(f"  Strict action match:   {strict_correct}/{n}  = {strict_correct/n:.1%}")
    print(f"  Relaxed action match:  {relaxed_correct}/{n}  = {relaxed_correct/n:.1%}")
    print(f"\n  Per-class strict (action == ideal):")
    for s, (sc, total) in by_status.items():
        if total > 0:
            print(f"    {s:<15}  {sc}/{total} = {sc/total:.1%}  (ideal: '{expected_action[s]}')")
    print(f"\n  Confusion (true → predicted action):")
    for s in expected_action:
        cells = ", ".join(f"{a}:{confusion.get((s,a),0)}"
                          for a in ["Pause", "Pivot", "Continue", "Scale"])
        print(f"    {s:<15}  {cells}")

    # ---------- 4. Recommender quality ----------
    section("4. RECOMMENDER QUALITY (top-5 similar)")
    # For each creative, fraction of its top-5 neighbors with the SAME status.
    # And mean perf delta (do top-performer neighbors get higher perf_score?).
    sample_for_rec = p._master_df.sample(min(150, n_total), random_state=7)
    same_status_count = []
    perf_delta_for_topperf = []
    for _, row in sample_for_rec.iterrows():
        cid = int(row["creative_id"])
        true = row["creative_status"]
        try:
            sims = p.find_similar(cid, k=5)
        except Exception:
            continue
        if not sims:
            continue
        same = sum(1 for s in sims if s["creative_status"] == true) / len(sims)
        same_status_count.append(same)
        if true == "top_performer":
            mean_neighbor_perf = np.mean([s["perf_score"] for s in sims])
            perf_delta_for_topperf.append(mean_neighbor_perf - float(row["perf_score"]))

    print(f"  Mean same-status fraction in top-5: {np.mean(same_status_count):.2%}")
    print(f"  ↑ Random baseline (status prior):   "
          f"{np.mean([(p._master_df['creative_status']==s).mean()**2 for s in expected_action])**0.5:.2%} or so")
    if perf_delta_for_topperf:
        print(f"  For top_performer queries: mean(neighbor_perf - own_perf) = "
              f"{np.mean(perf_delta_for_topperf):+.3f}  "
              f"(closer to 0 = retrieving similarly-performant peers)")

    # ---------- 5. Cluster quality ----------
    section("5. CLUSTER QUALITY")
    cdf = pd.read_parquet("outputs/clusters/labels.parquet").merge(
        p._master_df[["creative_id", "vertical", "creative_status"]], on="creative_id"
    )
    sized = cdf[cdf.cluster_id >= 0]
    purities = []
    for cid_, sub in sized.groupby("cluster_id"):
        purities.append(sub["vertical"].value_counts(normalize=True).iloc[0])
    print(f"  Clusters:           {sized['cluster_id'].nunique()}")
    print(f"  Mean cluster size:  {sized.groupby('cluster_id').size().mean():.1f}")
    print(f"  Median size:        {int(sized.groupby('cluster_id').size().median())}")
    print(f"  Vertical purity:    mean={np.mean(purities):.2%}  min={np.min(purities):.2%}")
    print(f"  Noise points:      {(cdf.cluster_id == -1).sum()} / {len(cdf)}  "
          f"({100*(cdf.cluster_id == -1).mean():.1f}%)")

    # ---------- 6. Sample explanations ----------
    section("6. SAMPLE EXPLANATIONS (one per status class)")
    for s in ["top_performer", "stable", "fatigued", "underperformer"]:
        sub = p._master_df[p._master_df["creative_status"] == s]
        cid = int(sub.sample(1, random_state=11).iloc[0]["creative_id"])
        e = p.explain(cid)
        h = e["health"]
        print(f"\n  cid={cid}  true={s}  →  {h['action']} (Health {h['health_score']}/100)")
        print(f"    {e['headline']}")
        if e["why_it_works"]:
            print(f"    Wins: {' | '.join(e['why_it_works'][:2])}")
        if e["what_to_watch"]:
            print(f"    Risks: {' | '.join(e['what_to_watch'][:2])}")
        if e["counterfactuals"]:
            print(f"    Try: {e['counterfactuals'][0]['advice']}")
        c = p.cluster_info(cid)
        print(f"    Cluster: {c['cluster_name']}  (n={c['n_members']})")

    print("\n" + "=" * 78)
    print("  EVAL DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
