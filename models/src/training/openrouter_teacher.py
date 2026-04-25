"""
OpenRouter-based teacher labeler.

Uses OpenRouter's API (OpenAI-compatible) with a vision model to generate
structured JSON pseudo-labels for each ad creative. These labels are the
teacher's "ground truth" for SDFT on-policy distillation.

Recommended models (cheapest → most capable):
  - google/gemini-flash-1.5         (~$0.075/1M tokens, fast)
  - google/gemini-2.0-flash-001     (~$0.10/1M tokens, best value)
  - openai/gpt-4o-mini              (~$0.15/1M tokens, great JSON)
  - anthropic/claude-3-5-haiku      (~$0.25/1M tokens, reliable)
  - qwen/qwen2-vl-72b-instruct      (~$0.40/1M tokens, best at ad analysis)

Cost estimate for 1,080 creatives: ~$0.10–$1.50 depending on model.
"""

import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are an expert ad creative performance analyst. "
    "Given a creative image and its performance metadata, produce a precise JSON analysis. "
    "Be specific about visual elements you can actually see in the image."
)

USER_TEMPLATE = """\
Analyze this ad creative image and explain its performance.

Creative metadata:
- Format: {format} | Theme: {theme} | Hook: {hook_type}
- CTA: {cta_text} | Tone: {emotional_tone} | Color: {dominant_color}
- Text density: {text_density:.2f} | Readability: {readability_score:.2f}
- Novelty: {novelty_score:.2f} | Motion: {motion_score:.2f}
- Brand visibility: {brand_visibility_score:.2f} | Clutter: {clutter_score:.2f}
- Faces: {faces_count} | Products: {product_count}
- Has discount badge: {has_discount_badge} | Has gameplay: {has_gameplay}
- Duration: {duration_sec}s | Copy length: {copy_length_chars} chars

Performance data:
- Status: {creative_status}
- Overall CTR: {overall_ctr:.4f} | Overall IPM: {overall_ipm:.3f}
- CTR decay: {ctr_decay_pct:.1%} | First 7d CTR: {first_7d_ctr:.4f} | Last 7d CTR: {last_7d_ctr:.4f}
- Vertical: {vertical} | Objective: {objective} | KPI goal: {kpi_goal}

Respond ONLY with this JSON (no markdown, no explanation outside the JSON):
{{
  "performance_summary": "<1-2 sentences: why does this creative perform as it does? Be specific.>",
  "visual_strengths": ["<strength 1>", "<strength 2>"],
  "visual_weaknesses": ["<weakness 1>"],
  "fatigue_risk_reason": "<why is/isn't this creative at risk of fatigue?>",
  "top_recommendation": "<single most impactful change to improve this creative>"
}}"""

METADATA_FIELDS = [
    "format", "theme", "hook_type", "cta_text", "emotional_tone", "dominant_color",
    "text_density", "readability_score", "novelty_score", "motion_score",
    "brand_visibility_score", "clutter_score", "faces_count", "product_count",
    "has_discount_badge", "has_gameplay", "duration_sec", "copy_length_chars",
    "creative_status", "overall_ctr", "overall_ipm", "ctr_decay_pct",
    "first_7d_ctr", "last_7d_ctr", "vertical", "objective", "kpi_goal",
]


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _safe_format(row: Dict) -> Dict:
    safe = {}
    for k in METADATA_FIELDS:
        v = row.get(k, "unknown")
        if isinstance(v, float):
            safe[k] = v if not (v != v) else 0.0  # handle NaN
        elif v is None or str(v) == "nan":
            safe[k] = "unknown"
        else:
            safe[k] = v
    return safe


def _parse_json(text: str) -> Optional[Dict]:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end]) if start >= 0 else None
    except json.JSONDecodeError:
        return None


def _validate(label: Optional[Dict]) -> bool:
    if not isinstance(label, dict):
        return False
    required = ["performance_summary", "visual_strengths", "visual_weaknesses",
                "fatigue_risk_reason", "top_recommendation"]
    return (
        all(k in label for k in required)
        and len(label.get("performance_summary", "")) > 20
        and isinstance(label.get("visual_strengths"), list)
    )


class OpenRouterTeacher:
    """
    Calls OpenRouter vision API to generate teacher pseudo-labels.
    Supports resuming interrupted runs.
    """

    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-001",
        api_key: Optional[str] = None,
        requests_per_minute: int = 30,
        max_retries: int = 3,
    ):
        self.model = model
        self.rpm = requests_per_minute
        self.max_retries = max_retries
        self._delay = 60.0 / requests_per_minute

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENROUTER_API_KEY env var or pass api_key= argument.\n"
                "Get a key at: https://openrouter.ai/keys"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://smadex-creative-intelligence",
                "X-Title": "Smadex Creative Intelligence",
            },
        )

    def label_one(self, row: Dict, image_path: Path) -> Optional[Dict]:
        img_b64 = _encode_image(image_path)
        safe_row = _safe_format(row)

        # Build prompt — handle float formatting carefully
        try:
            prompt = USER_TEMPLATE.format(**safe_row)
        except (KeyError, ValueError):
            prompt = USER_TEMPLATE.format(**{k: safe_row.get(k, "unknown") for k in METADATA_FIELDS})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    max_tokens=400,
                    temperature=0.1,
                )
                text = response.choices[0].message.content or ""
                label = _parse_json(text)
                if _validate(label):
                    return label
                if attempt < self.max_retries - 1:
                    time.sleep(2)
            except Exception as e:
                wait = 2 ** attempt * 2
                print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return None

    def label_all(
        self,
        master_df,
        asset_dir: Path,
        output_path: Path,
        resume: bool = True,
        verbose: bool = True,
        max_workers: int = 1,
    ) -> List[Dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids: set = set()

        if resume and output_path.exists():
            with open(output_path) as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line)["creative_id"])
                    except Exception:
                        pass
            if verbose and existing_ids:
                print(f"Resuming: {len(existing_ids)} already labeled, "
                      f"{len(master_df) - len(existing_ids)} remaining.")

        # Build the work queue
        todo: List[Dict] = []
        for _, row in master_df.iterrows():
            cid = int(row["creative_id"])
            if cid in existing_ids:
                continue
            if not (asset_dir / f"creative_{cid}.png").exists():
                continue
            todo.append({"cid": cid, "row": row.to_dict()})

        results: List[Dict] = []
        n_total = len(master_df)
        n_done = len(existing_ids)
        write_lock = threading.Lock()
        out_f = open(output_path, "a")

        def _worker(item: Dict):
            cid = item["cid"]
            label = self.label_one(item["row"], asset_dir / f"creative_{cid}.png")
            return cid, label

        try:
            if max_workers <= 1:
                for item in todo:
                    cid, label = _worker(item)
                    n_done += 1
                    if label is not None:
                        record = {"creative_id": cid, "model": self.model, **label}
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        results.append(record)
                    if verbose and (n_done % 20 == 0 or n_done == n_total):
                        print(f"  [{n_done}/{n_total}] creative_{cid} → {'OK' if label else 'FAILED'}")
                    time.sleep(self._delay)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_worker, item): item for item in todo}
                    for fut in as_completed(futures):
                        cid, label = fut.result()
                        with write_lock:
                            n_done += 1
                            if label is not None:
                                record = {"creative_id": cid, "model": self.model, **label}
                                out_f.write(json.dumps(record) + "\n")
                                out_f.flush()
                                results.append(record)
                            if verbose and (n_done % 20 == 0 or n_done == n_total):
                                print(f"  [{n_done}/{n_total}] {'OK' if label else 'FAILED'}", flush=True)
        finally:
            out_f.close()
        return results
