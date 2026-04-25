"""
OpenRouter-based STRUCTURED RUBRIC extractor (offline, one-time).

Difference vs `openrouter_teacher.py`:
  - teacher: free-form text labels for VLM SDFT distillation
  - this:    15 numeric rubric scores per creative, cached to parquet

The rubric becomes part of the Genome Vector. It is computed ONCE during
training and read from disk at inference — no LLM call at runtime.

Cost: ~$0.10–$1.00 for all 1,080 creatives (depending on model).
"""

import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


# ---------- Rubric definition ----------
# Each dim is on 0–10 (visual qualities) or a small integer count.
# Keep this list stable — column names get baked into the trained model.
# Each field has anchored description: 0 / 5 / 10 examples to prevent median-defaulting.
RUBRIC_FIELDS: List[Tuple[str, str]] = [
    ("hook_clarity",
     "How fast does the value-prop land? "
     "0=text-wall, no clear message; 3=value-prop buried; 5=decipherable in 2-3s; "
     "8=clear in <1s; 10=instantly obvious from headline+image."),
    ("cta_prominence",
     "Visual prominence of the call-to-action. "
     "0=no visible CTA at all; 3=CTA tiny / corner; 5=CTA present but blends in; "
     "8=clearly highlighted; 10=CTA dominates the composition."),
    ("cta_contrast",
     "Contrast of CTA against its background (color/luminance). "
     "0=no CTA or invisible; 3=low contrast (similar tones); 5=moderate; "
     "8=high contrast (e.g., neon-on-dark); 10=maximum (white-on-black or complementary)."),
    ("color_vibrancy",
     "Saturation+brightness of dominant palette. "
     "0=grayscale/desaturated; 3=muted earth tones; 5=balanced; 8=vivid; 10=neon/maxed-out."),
    ("color_warmth",
     "Warm vs cool palette. "
     "0=deep cool (blue/black); 3=cool-leaning; 5=neutral; 8=warm-leaning; 10=hot (red/orange/yellow)."),
    ("text_density_visual",
     "Pixel area covered by text vs total area. "
     "0=no text; 3=minimal headline; 5=balanced text+image; 8=text-heavy; 10=copy dominates."),
    ("face_count_visual",
     "Count of distinct human faces clearly visible. Score is the literal integer count (cap at 10)."),
    ("product_focus",
     "How clearly the product is the visual subject. "
     "0=product absent / lifestyle only; 3=product in background; 5=product alongside other elements; "
     "8=product centered; 10=product fills the frame (hero shot)."),
    ("scene_realism",
     "Photo-real vs illustrated/UI. "
     "0=pure illustration / vector / UI mock; 3=stylized illustration; 5=mixed photo+graphic; "
     "8=photo-leaning; 10=pure photograph."),
    ("emotion_intensity",
     "Emotional charge of the imagery. "
     "0=clinical/neutral; 3=mild; 5=clearly emotive; 8=high-energy or strongly evocative; 10=intense (joy/awe/urgency)."),
    ("composition_balance",
     "Quality of layout / visual hierarchy. "
     "0=chaotic, no focal point; 3=cluttered; 5=acceptable; 8=clean grid / clear hierarchy; 10=textbook layout."),
    ("brand_visibility",
     "Brand logo/name visibility. "
     "0=no brand visible; 3=tiny logo in corner; 5=logo present but secondary; 8=clearly placed; 10=brand dominates."),
    ("urgency_signal",
     "Limited-time / scarcity / discount signaling. "
     "0=none; 3=subtle hint (e.g., 'New'); 5=clear offer mentioned; 8=prominent discount badge; 10=multiple urgency cues (timer + %off + 'now')."),
    ("playfulness",
     "Playful/fun tone vs serious. "
     "0=corporate/clinical; 3=measured; 5=neutral; 8=fun/casual; 10=cartoony/whimsical."),
    ("novelty_visual",
     "How fresh/non-template the design feels. "
     "0=generic stock template; 3=common pattern; 5=competent but familiar; 8=distinctive; 10=highly original / unusual."),
]
RUBRIC_NAMES: List[str] = [n for n, _ in RUBRIC_FIELDS]


# ---------- Prompt v2 (anchored, anti-default) ----------
SYSTEM_PROMPT = (
    "You are a senior ad-creative analyst. You score creatives on a fixed numeric rubric. "
    "Score ONLY from the image. Critically: distribute scores across the full 0-10 range. "
    "Most creatives are NOT mid-tier on every axis — some are 1, some are 9. "
    "If you would default to 5 across the board, you are not looking carefully enough. "
    "For face_count_visual: report the literal integer count of visible faces (often 0)."
)

USER_PROMPT = (
    "Score this ad creative on the rubric below. "
    "Use the anchor levels (0 / 3 / 5 / 8 / 10) as calibration — pick the closest integer 0-10. "
    "Output ONLY a JSON object — no markdown fences, no commentary.\n\n"
    "Rubric:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in RUBRIC_FIELDS)
    + "\n\nReturn JSON shaped exactly like (replace zeros with your scores):\n"
    + json.dumps({n: 0 for n in RUBRIC_NAMES}, indent=2)
)


@dataclass
class RubricResult:
    creative_id: int
    scores: Dict[str, int]
    model: str


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json(text: str) -> Optional[Dict]:
    text = text.strip()
    if text.startswith("```"):
        # strip ``` fences
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end]) if start >= 0 else None
    except json.JSONDecodeError:
        return None


def _coerce_scores(raw: Optional[Dict]) -> Optional[Dict[str, int]]:
    if not isinstance(raw, dict):
        return None
    out: Dict[str, int] = {}
    for n in RUBRIC_NAMES:
        v = raw.get(n)
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        out[n] = max(0, min(10, v))
    return out


class OpenRouterRubric:
    def __init__(
        self,
        model: str = "google/gemini-2.0-flash-001",
        api_key: Optional[str] = None,
        requests_per_minute: int = 30,
        max_retries: int = 3,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENROUTER_API_KEY env var or pass api_key=. "
                "Get one at https://openrouter.ai/keys"
            )
        self.model = model
        self.max_retries = max_retries
        self._delay = 60.0 / max(1, requests_per_minute)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://smadex-creative-intelligence",
                "X-Title": "Smadex Creative Genome",
            },
        )

    def score_one(self, image_path: Path) -> Optional[Dict[str, int]]:
        img_b64 = _encode_image(image_path)
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
                                {"type": "text", "text": USER_PROMPT},
                            ],
                        },
                    ],
                    max_tokens=300,
                    temperature=0.0,
                )
                text = response.choices[0].message.content or ""
                scores = _coerce_scores(_parse_json(text))
                if scores is not None:
                    return scores
            except Exception as e:
                wait = 2 ** attempt * 2
                print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return None

    def score_all(
        self,
        creative_ids: List[int],
        asset_dir: Path,
        output_path: Path,
        resume: bool = True,
        verbose: bool = True,
        max_workers: int = 1,
    ) -> List[RubricResult]:
        """Streams JSONL to `output_path` so partial progress survives crashes.
        max_workers > 1 enables concurrent API calls (recommended: 32-64)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing: set = set()
        if resume and output_path.exists():
            with open(output_path) as f:
                for line in f:
                    try:
                        existing.add(int(json.loads(line)["creative_id"]))
                    except Exception:
                        pass
            if verbose and existing:
                print(f"Resuming: {len(existing)} already scored, "
                      f"{len(creative_ids) - len(existing)} remaining.")

        todo: List[int] = []
        for cid in creative_ids:
            cid = int(cid)
            if cid in existing:
                continue
            if not (asset_dir / f"creative_{cid}.png").exists():
                continue
            todo.append(cid)

        results: List[RubricResult] = []
        n_total = len(creative_ids)
        n_done = len(existing)
        write_lock = threading.Lock()
        out_f = open(output_path, "a")

        def _worker(cid: int) -> Tuple[int, Optional[Dict[str, int]]]:
            scores = self.score_one(asset_dir / f"creative_{cid}.png")
            return cid, scores

        try:
            if max_workers <= 1:
                for cid in todo:
                    _, scores = _worker(cid)
                    n_done += 1
                    if scores is not None:
                        record = {"creative_id": cid, "model": self.model, **scores}
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        results.append(RubricResult(cid, scores, self.model))
                    if verbose and (n_done % 20 == 0 or n_done == n_total):
                        print(f"  [{n_done}/{n_total}] {'OK' if scores else 'FAILED'}")
                    time.sleep(self._delay)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_worker, cid): cid for cid in todo}
                    for fut in as_completed(futures):
                        cid, scores = fut.result()
                        with write_lock:
                            n_done += 1
                            if scores is not None:
                                record = {"creative_id": cid, "model": self.model, **scores}
                                out_f.write(json.dumps(record) + "\n")
                                out_f.flush()
                                results.append(RubricResult(cid, scores, self.model))
                            if verbose and (n_done % 20 == 0 or n_done == n_total):
                                print(f"  [{n_done}/{n_total}] {'OK' if scores else 'FAILED'}", flush=True)
        finally:
            out_f.close()
        return results
