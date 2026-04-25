"""
Generate human-readable cluster names from modal metadata — deterministic, no LLM.

Each cluster is named like: "Gaming · Interstitial · Fantasy (purple)" using the
most common (vertical, format, theme, dominant_color) tuple. Output is cached
into outputs/clusters/cluster_names.parquet for instant lookup at demo time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

CLUSTER_LABELS = "outputs/clusters/labels.parquet"
CLUSTER_NAMES = "outputs/clusters/cluster_names.parquet"


def title(s: str) -> str:
    return str(s).replace("_", " ").title() if s else ""


def main():
    clusters = pd.read_parquet(CLUSTER_LABELS)
    summary = pd.read_csv("../creative_summary.csv")[
        ["creative_id", "vertical", "format", "dominant_color", "theme",
         "emotional_tone", "hook_type", "cta_text"]
    ]
    df = clusters.merge(summary, on="creative_id", how="left")

    rows = []
    for cid, sub in df.groupby("cluster_id"):
        if cid == -1:
            rows.append({
                "cluster_id": -1,
                "name": "Outliers",
                "size": len(sub),
                "vertical": "mixed",
                "format": "mixed",
                "theme": "mixed",
                "dominant_color": "mixed",
                "tone": "mixed",
            })
            continue
        v = sub["vertical"].mode().iloc[0]
        fmt = sub["format"].mode().iloc[0]
        theme = sub["theme"].mode().iloc[0]
        color = sub["dominant_color"].mode().iloc[0]
        tone = sub["emotional_tone"].mode().iloc[0]
        # purity score = how dominant the modal vertical is
        purity = sub["vertical"].value_counts(normalize=True).iloc[0]
        name = f"{title(v)} · {title(fmt)} · {title(theme)} ({color})"
        rows.append({
            "cluster_id": int(cid),
            "name": name,
            "size": len(sub),
            "vertical": v,
            "format": fmt,
            "theme": theme,
            "dominant_color": color,
            "tone": tone,
            "purity": round(purity, 2),
        })

    out = pd.DataFrame(rows).sort_values("cluster_id")
    Path(CLUSTER_NAMES).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CLUSTER_NAMES, index=False)

    print(f"Named {len(out)} clusters → {CLUSTER_NAMES}\n")
    for _, r in out.iterrows():
        print(f"  cid={r['cluster_id']:>3}  n={r['size']:>3}  purity={r.get('purity', '-'):>4}  {r['name']}")


if __name__ == "__main__":
    main()
