"""
Loads precomputed natural-language teacher annotations from JSONL.

These are the OpenRouter-generated explanations attached to each creative
(performance_summary, visual_strengths, visual_weaknesses, fatigue_risk_reason,
top_recommendation). Read once at startup, served from RAM.

No LLM at runtime — same speed-first principle as the rubric features.
"""
import json
from pathlib import Path
from typing import Dict, Optional


def load_annotations(jsonl_path: Path) -> Dict[int, Dict]:
    """Returns {creative_id: annotation_dict}. Empty dict if the file is missing."""
    out: Dict[int, Dict] = {}
    p = Path(jsonl_path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            try:
                rec = json.loads(line)
                cid = int(rec["creative_id"])
                out[cid] = {
                    "performance_summary": rec.get("performance_summary", ""),
                    "visual_strengths": rec.get("visual_strengths", []),
                    "visual_weaknesses": rec.get("visual_weaknesses", []),
                    "fatigue_risk_reason": rec.get("fatigue_risk_reason", ""),
                    "top_recommendation": rec.get("top_recommendation", ""),
                    "source_model": rec.get("model", "unknown"),
                }
            except Exception:
                pass
    return out


def get_annotation(annotations: Dict[int, Dict], creative_id: int) -> Optional[Dict]:
    return annotations.get(int(creative_id))
