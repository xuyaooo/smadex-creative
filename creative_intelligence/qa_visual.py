"""Generate a per-creative diagnostic JSON for human eyeball comparison
against the actual PNG asset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.inference.pipeline import CreativeIntelligencePipeline

P = CreativeIntelligencePipeline("config.yaml")
P._ensure_models()

# Pull rubric values for everyone (cached parquet)
rubric_df = pd.read_parquet("outputs/rubric/rubric_scores.parquet").set_index("creative_id")

# Pick one diverse creative per status class
picks = []
for s in ["top_performer", "stable", "fatigued", "underperformer"]:
    sub = P._master_df[P._master_df["creative_status"] == s]
    cid = int(sub.sample(1, random_state=11).iloc[0]["creative_id"])
    picks.append((s, cid))

print("Selected creatives:")
for s, cid in picks:
    print(f"  {s:<15}  cid={cid}")

# For each pick, dump everything
for true_status, cid in picks:
    row = P._master_df[P._master_df["creative_id"] == cid].iloc[0]
    h = P.health_score(cid)
    e = P.explain(cid)
    sims = P.find_similar(cid, k=5)
    cluster = P.cluster_info(cid)

    rubric_scores = {}
    if cid in rubric_df.index:
        rubric_scores = {k: int(v) for k, v in rubric_df.loc[cid].items()}

    out = {
        "creative_id": cid,
        "image_path": f"../assets/creative_{cid}.png",
        "ground_truth": {
            "creative_status": str(row["creative_status"]),
            "vertical": str(row.get("vertical", "?")),
            "format": str(row.get("format", "?")),
            "theme": str(row.get("theme", "?")),
            "dominant_color_metadata": str(row.get("dominant_color", "?")),
            "emotional_tone_metadata": str(row.get("emotional_tone", "?")),
            "headline": str(row.get("headline", "?")),
            "subhead": str(row.get("subhead", "?")),
            "cta_text": str(row.get("cta_text", "?")),
            "has_discount_badge": int(row.get("has_discount_badge", 0)),
            "has_price": int(row.get("has_price", 0)),
            "faces_count_metadata": int(row.get("faces_count", 0)),
            "perf_score": float(row.get("perf_score", 0)),
            "fatigue_day": (None if pd.isna(row.get("fatigue_day")) else int(row.get("fatigue_day"))),
        },
        "rubric_scores_from_LLM": rubric_scores,
        "predictions": {
            "predicted_status_via_action": h["action"],
            "health_score": h["health_score"],
            "predicted_perf": h["predicted_perf"],
            "fatigue_probability": round(100 * h["components"]["fatigue_resistance"] / 100, 4),
            "fatigue_resistance": h["components"]["fatigue_resistance"],
        },
        "explanation": {
            "headline": e["headline"],
            "why_it_works": e["why_it_works"],
            "what_to_watch": e["what_to_watch"],
            "rubric_callouts": e["rubric_callouts"],
            "counterfactuals": [cf["advice"] for cf in e["counterfactuals"]],
        },
        "cluster": {
            "name": cluster["cluster_name"],
            "size": cluster["n_members"],
        },
        "top_5_similar": [
            {
                "creative_id": s["creative_id"],
                "similarity": s["similarity"],
                "creative_status": s["creative_status"],
                "perf_score": s["perf_score"],
            } for s in sims
        ],
    }
    out_path = Path(f"qa_visual_{cid}.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")

print("\nAll diagnostic JSONs written. Next step: read each image + JSON pair side by side.")
