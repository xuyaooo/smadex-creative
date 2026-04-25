"""QA integration script — exercises the four demo-facing pipeline methods
against three creatives that span different true statuses. Also verifies the
diversify=True branch returns 5 distinct creative_ids.

Run with:
    python3 tests/qa_pipeline_integration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.inference.pipeline import CreativeIntelligencePipeline


def main() -> int:
    p = CreativeIntelligencePipeline("config.yaml")
    p._ensure_models()

    # Pick one creative per true status
    df = p._master_df
    targets = []
    for s in ["top_performer", "stable", "fatigued", "underperformer"]:
        sub = df[df["creative_status"] == s]
        if not sub.empty:
            targets.append((s, int(sub.iloc[0]["creative_id"])))

    failed = []
    for status, cid in targets:
        print(f"\n--- creative {cid} (true={status}) ---")
        try:
            h = p.health_score(cid)
            assert isinstance(h, dict) and "health_score" in h and "action" in h
            print(f"  health_score: {h['health_score']} action={h['action']}")
        except Exception as e:
            failed.append(("health_score", cid, repr(e)))

        try:
            e = p.explain(cid)
            assert "headline" in e and "counterfactuals" in e and "shap_top_pos" in e
            print(f"  explain: {e['headline'][:60]}...")
        except Exception as ex:
            failed.append(("explain", cid, repr(ex)))

        try:
            sims = p.find_similar(cid, k=5)
            assert isinstance(sims, list) and len(sims) == 5
            for s in sims:
                assert "creative_id" in s and isinstance(s["creative_id"], int)
                assert s["creative_id"] != cid
            print(f"  find_similar: {[s['creative_id'] for s in sims]}")
        except Exception as ex:
            failed.append(("find_similar", cid, repr(ex)))

        try:
            c = p.cluster_info(cid)
            assert "cluster_id" in c and "cluster_name" in c
            print(f"  cluster_info: id={c['cluster_id']} name={c['cluster_name']}")
        except Exception as ex:
            failed.append(("cluster_info", cid, repr(ex)))

        try:
            sims_d = p.find_similar(cid, k=5, diversify=True)
            ids = [s["creative_id"] for s in sims_d]
            assert len(sims_d) == 5
            assert len(set(ids)) == 5, f"diversify duplicates: {ids}"
            assert cid not in ids, "query leaked into result"
            print(f"  find_similar(diversify=True): {ids}")
        except Exception as ex:
            failed.append(("find_similar(diversify)", cid, repr(ex)))

    if failed:
        print("\nFAILURES:")
        for f in failed:
            print(" ", f)
        return 1
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
