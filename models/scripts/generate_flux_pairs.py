"""
Build the (source, ensemble-brief, target) training corpus for Flux edit.

For every weak / fatigued / underperformer creative we:
  1. Run the trained tabular ensemble to extract its diagnosis:
       - predicted_status, health_score, class probabilities
       - top-3 single-feature counterfactual lifts
       - data-grounded palette for its vertical (from build_palette_lookup)
  2. Compose a "rebuild brief" string from the diagnosis.
  3. Send (source image + brief) to Gemini 2.5 Flash Image (Nano Banana) over
     the OpenRouter API.
  4. Re-score the returned image with the same tabular ensemble. Keep the
     pair only if post-edit health ≥ POSITIVE_THRESHOLD (default 75) — clean
     positives only.
  5. Persist (source_path, brief, target_path, score_lift) to a JSONL manifest
     and the generated images to disk.

Output layout:
    models/outputs/flux_pairs/
        manifest.jsonl
        images/
            <creative_id>__source.png
            <creative_id>__target.png

Run:
    cd models && PYTHONPATH=$PWD python3 scripts/generate_flux_pairs.py \
        --max 200 \
        --threshold 75 \
        --model google/gemini-2.5-flash-image
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = REPO / "data"
ASSETS = DATA / "assets"
sys.path.insert(0, str(ROOT))

from src.inference.pipeline import CreativeIntelligencePipeline  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-image"
OUT_DIR = ROOT / "outputs" / "flux_pairs"


# ---------------------------------------------------------------------------
def encode_image(p: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"


def decode_image(data_url: str, dst: Path) -> None:
    """Strip the data-url prefix and write the bytes."""
    _, payload = data_url.split(",", 1)
    dst.write_bytes(base64.b64decode(payload))


def load_palette() -> dict:
    """Cached per-vertical palette — emit empty if not built yet."""
    p = REPO / "front/public/data/palettes.json"
    if not p.exists():
        print("WARN: palettes.json missing — run build_palette_lookup.py first")
        return {}
    return json.loads(p.read_text()).get("per_vertical", {})


def build_brief(row: Dict[str, Any], pipeline_out: Dict[str, Any], palette: list[dict]) -> str:
    cf = pipeline_out.get("counterfactuals", [])[:3]
    cf_block = "\n".join([
        f"- change {c['feat']} from \"{c['from']}\" to \"{c['to']}\" → score ≈ {c['score']:.0f}"
        for c in cf
    ]) or "  (no high-confidence counterfactuals)"
    palette_block = "\n".join([f"- {c['hex'].upper()} ({c['label']})" for c in palette[:5]]) or "  (no palette)"

    return f"""# Creative-rebuild brief (ensemble-driven)

## Context
- Vertical: {row['vertical']}
- Format: {row['format']}
- Theme: {row.get('theme', 'unknown')}
- Predicted status: {pipeline_out['predicted_status']}
- Health score: {pipeline_out['health_score']:.0f}/100

## Counterfactual lifts (single-feature swaps the ensemble suggests)
{cf_block}

## Palette (mandatory — use these hexes only, plus #ffffff/#000000 for text)
{palette_block}

## Output
Return ONE polished mobile ad creative image that applies the suggested lifts.
Preserve brand identity, the primary product, and the overall scene.
CTA contrast ≥ 4.5:1. Crisp typography. No AI artifacts.
"""


def call_nano_banana(api_key: str, model: str, image_path: Path, brief: str) -> Optional[bytes]:
    """Returns the raw bytes of the edited image, or None if the call failed."""
    import requests

    img = encode_image(image_path)
    res = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://creative-ai-companion",
            "X-Title": "Flux pairs builder",
        },
        json={
            "model": model,
            "modalities": ["text", "image"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img}},
                        {"type": "text", "text": brief},
                    ],
                }
            ],
        },
        timeout=120,
    )
    if res.status_code != 200:
        print(f"  ! API {res.status_code}: {res.text[:200]}")
        return None
    body = res.json()
    msg = body.get("choices", [{}])[0].get("message", {})
    images = msg.get("images") or []
    for im in images:
        url = (im.get("image_url") or {}).get("url") or im.get("url")
        if isinstance(url, str) and url.startswith("data:image"):
            return base64.b64decode(url.split(",", 1)[1])
    # Walk multimodal content array as a fallback
    for part in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
        if part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if isinstance(url, str) and url.startswith("data:image"):
                return base64.b64decode(url.split(",", 1)[1])
    return None


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=200, help="max creatives to process")
    parser.add_argument("--threshold", type=int, default=75, help="min post-edit health to keep the pair")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--statuses", type=str, default="fatigued,underperformer,stable",
                        help="comma-list of statuses to source weak creatives from")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY (Nano Banana lives on OpenRouter).")

    print("Booting tabular ensemble (used to score before & after)…")
    pipeline = CreativeIntelligencePipeline(str(ROOT / "config.yaml"))
    pipeline._ensure_models()
    palette_lookup = load_palette()

    summary = pd.read_csv(DATA / "creative_summary.csv")
    statuses = {s.strip() for s in args.statuses.split(",")}
    weak = summary[summary["creative_status"].isin(statuses)].head(args.max)
    print(f"Generating pairs for {len(weak)} creatives in [{', '.join(sorted(statuses))}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.jsonl"
    kept = dropped = 0

    with open(manifest_path, "a") as mf:
        for _, row in weak.iterrows():
            cid = int(row["creative_id"])
            src_img = ASSETS / f"creative_{cid}.png"
            if not src_img.exists():
                continue

            # Pre-edit diagnosis from the ensemble.
            try:
                pre = pipeline.health_score(cid)
                pre.update(pipeline.explain(cid))     # adds counterfactuals
            except Exception as e:
                print(f"  cid={cid} skip (ensemble): {e}")
                continue

            palette = palette_lookup.get(row["vertical"], [])
            brief = build_brief(row.to_dict(), pre, palette)

            # Call Nano Banana.
            print(f"→ cid={cid}  pre_health={pre['health_score']:.0f}", end=" ")
            t0 = time.time()
            bytes_ = call_nano_banana(api_key, args.model, src_img, brief)
            dt = time.time() - t0
            if bytes_ is None:
                print("✗ no image")
                dropped += 1
                continue

            tgt_img = OUT_DIR / "images" / f"{cid}__target.png"
            src_dst = OUT_DIR / "images" / f"{cid}__source.png"
            tgt_img.write_bytes(bytes_)
            src_dst.write_bytes(src_img.read_bytes())

            # Post-edit re-score by swapping the image path under the same metadata.
            post_health = pre["health_score"]
            try:
                post = pipeline.score_new_creative(str(tgt_img), row.to_dict())
                post_health = float(post.get("perf_score", pre["health_score"]))
            except Exception as e:
                print(f"  cid={cid} re-score failed: {e}")

            lift = post_health - pre["health_score"]
            keep = post_health >= args.threshold

            print(f"  post={post_health:.0f}  lift={lift:+.1f}  {'✓ kept' if keep else '✗ dropped'}  {dt:.1f}s")

            if keep:
                mf.write(json.dumps({
                    "creative_id": cid,
                    "vertical": row["vertical"],
                    "format": row["format"],
                    "source": str(src_dst.relative_to(REPO)),
                    "target": str(tgt_img.relative_to(REPO)),
                    "brief": brief,
                    "pre_health": pre["health_score"],
                    "post_health": post_health,
                    "lift": lift,
                }) + "\n")
                mf.flush()
                kept += 1
            else:
                tgt_img.unlink(missing_ok=True)
                src_dst.unlink(missing_ok=True)
                dropped += 1

    print(f"\nDone. Kept {kept} pairs, dropped {dropped}. Manifest: {manifest_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
