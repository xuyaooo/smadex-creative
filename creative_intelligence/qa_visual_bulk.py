"""Larger manual-QA sweep: 16 creatives, agreement table for dataset vs LLM rubric vs predictions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.inference.pipeline import CreativeIntelligencePipeline

P = CreativeIntelligencePipeline("config.yaml")
P._ensure_models()

rubric_df = pd.read_parquet("outputs/rubric/rubric_scores.parquet").set_index("creative_id")

# Pick 4 per status, diverse verticals if possible
PER_CLASS = 4
picks: list[tuple[str, int]] = []
rng = np.random.default_rng(42)
for s in ["top_performer", "stable", "fatigued", "underperformer"]:
    sub = P._master_df[P._master_df["creative_status"] == s].copy()
    # take 4 random, unique per status
    chosen = sub.sample(min(PER_CLASS, len(sub)), random_state=42)["creative_id"].astype(int).tolist()
    for cid in chosen:
        picks.append((s, cid))

print(f"Selected {len(picks)} creatives ({PER_CLASS} per status class).\n")

# Map status → expected action
EXPECTED_ACTION = {
    "top_performer":  "Scale",
    "stable":         "Continue",
    "fatigued":       "Pause",
    "underperformer": "Pivot",
}

# Build the agreement table
rows = []
for true_status, cid in picks:
    row = P._master_df[P._master_df["creative_id"] == cid].iloc[0]
    h = P.health_score(cid)

    # rubric scores from cached parquet
    rs = rubric_df.loc[cid] if cid in rubric_df.index else None

    # Data fields
    data_faces = int(row.get("faces_count", 0))
    data_discount = int(row.get("has_discount_badge", 0))
    data_price = int(row.get("has_price", 0))
    data_color = str(row.get("dominant_color", "?"))
    data_emotion = str(row.get("emotional_tone", "?"))

    # LLM-derived fields
    llm_faces = int(rs["face_count_visual"]) if rs is not None else -1
    llm_urgency = int(rs["urgency_signal"]) if rs is not None else -1
    llm_color_vibrancy = int(rs["color_vibrancy"]) if rs is not None else -1
    llm_emotion_intensity = int(rs["emotion_intensity"]) if rs is not None else -1
    llm_brand = int(rs["brand_visibility"]) if rs is not None else -1
    llm_cta_contrast = int(rs["cta_contrast"]) if rs is not None else -1
    llm_product_focus = int(rs["product_focus"]) if rs is not None else -1
    llm_realism = int(rs["scene_realism"]) if rs is not None else -1

    # Agreement decisions (boolean per row)
    # 1) faces: dataset says X faces. LLM detects 0 if no faces. Match = both >0 or both 0.
    faces_match = (data_faces > 0) == (llm_faces > 0)

    # 2) discount badge: dataset says 1 if creative has a SALE/discount pill.
    #    LLM urgency_signal >= 6 should trigger when the badge is visible.
    discount_match = (data_discount == 1) == (llm_urgency >= 6)

    # 3) price: dataset says 1 if a price is shown.
    #    LLM doesn't have a direct price dim, but urgency_signal correlates.
    #    Skip from the agreement metric — we'll just record.

    # 4) action: predicted vs expected
    action_match = h["action"] == EXPECTED_ACTION[true_status]

    rows.append({
        "cid": cid,
        "true_status": true_status,
        "predicted_action": h["action"],
        "action_match": action_match,
        "data_faces": data_faces,
        "llm_faces": llm_faces,
        "faces_match": faces_match,
        "data_discount": data_discount,
        "llm_urgency": llm_urgency,
        "discount_match": discount_match,
        "data_price": data_price,
        "data_color": data_color,
        "llm_vibrancy": llm_color_vibrancy,
        "data_emotion": data_emotion,
        "llm_emotion": llm_emotion_intensity,
        "llm_brand": llm_brand,
        "llm_cta_contrast": llm_cta_contrast,
        "llm_product_focus": llm_product_focus,
        "llm_realism": llm_realism,
        "perf_true": float(row["perf_score"]),
        "perf_pred": h["predicted_perf"],
        "health": h["health_score"],
    })

df = pd.DataFrame(rows)
print("=" * 110)
print("PER-CREATIVE AGREEMENT TABLE")
print("=" * 110)
short = df[[
    "cid", "true_status", "predicted_action", "action_match",
    "data_faces", "llm_faces", "faces_match",
    "data_discount", "llm_urgency", "discount_match",
    "perf_true", "perf_pred", "health",
]].copy()
print(short.to_string(index=False))

print("\n" + "=" * 110)
print("AGGREGATE AGREEMENT")
print("=" * 110)
print(f"  Action match (predicted action == expected for true status):")
print(f"    overall: {df['action_match'].sum()}/{len(df)} = {df['action_match'].mean():.0%}")
for s in ["top_performer", "stable", "fatigued", "underperformer"]:
    sub = df[df.true_status == s]
    print(f"    {s:<15}  {sub['action_match'].sum()}/{len(sub)} = {sub['action_match'].mean():.0%}")

print(f"\n  Dataset faces vs LLM faces:")
print(f"    Both saw faces or both saw none: {df['faces_match'].sum()}/{len(df)} = {df['faces_match'].mean():.0%}")
print(f"    Dataset says >0 faces but LLM says 0: {((df.data_faces>0) & (df.llm_faces==0)).sum()}/{len(df)}  ← these are likely DATA wrong")
print(f"    LLM says >0 faces but dataset says 0: {((df.llm_faces>0) & (df.data_faces==0)).sum()}/{len(df)}")

print(f"\n  Discount badge (dataset has_discount_badge==1) vs LLM urgency_signal>=6:")
print(f"    Match: {df['discount_match'].sum()}/{len(df)} = {df['discount_match'].mean():.0%}")
print(f"    Data says badge but LLM low urgency: {((df.data_discount==1) & (df.llm_urgency<6)).sum()}/{len(df)}")
print(f"    Data says no badge but LLM high urgency: {((df.data_discount==0) & (df.llm_urgency>=6)).sum()}/{len(df)}  ← LLM picking up other urgency cues (price, headline)")

print(f"\n  Predicted perf vs true perf:")
print(f"    Mean abs error: {(df.perf_pred - df.perf_true).abs().mean():.3f}")
print(f"    Sign of error: predicted - true average = {(df.perf_pred - df.perf_true).mean():+.3f}  (negative = underprediction)")

print(f"\n  Cases where data faces_count disagrees with LLM (dataset is suspect):")
disagree = df[df.data_faces > 0]
disagree = disagree[disagree.llm_faces == 0]
if len(disagree):
    print(disagree[["cid","true_status","data_faces","llm_faces","data_emotion"]].to_string(index=False))

# Save full results
df.to_csv("qa_visual_bulk.csv", index=False)
print(f"\nFull results saved to qa_visual_bulk.csv ({len(df)} rows × {len(df.columns)} cols)")
