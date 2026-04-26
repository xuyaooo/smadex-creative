"""
Build a data-grounded color palette per vertical from the actual top-performing
creatives' images.

Approach:
  1. Take every creative whose ground-truth status is `top_performer`.
  2. Open its PNG, downsample, and run k-means (k=3) over the pixels to
     extract its 3 dominant colors (HSV-weighted).
  3. Aggregate per vertical — keep the most-frequent / highest-saturation
     hexes across all top performers in that vertical.
  4. Also produce a per-(vertical × predicted_status) breakdown so the front
     end can pick a palette that matches the user's predicted bucket.

Output: front/public/data/palettes.json

Run:
    cd models && PYTHONPATH=$PWD python3 scripts/build_palette_lookup.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = REPO / "data"
ASSETS = DATA / "assets"
N_CLUSTERS = 6              # more clusters per image so we capture accents + brand
SAMPLE_SIZE = 128           # higher resolution sample → less aliasing
TOP_K_PER_VERTICAL = 6      # how many palette colors to keep per vertical
TOP_K_PER_BUCKET = 5

# Approximate role labels — we'll assign by HSV brightness/saturation
ROLE_LABELS = ["primary background", "accent · CTA", "secondary fill"]


def hex_of(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    rf, gf, bf = r / 255, g / 255, b / 255
    mx, mn = max(rf, gf, bf), min(rf, gf, bf)
    h = 0.0
    if mx != mn:
        if mx == rf:
            h = ((gf - bf) / (mx - mn)) % 6
        elif mx == gf:
            h = (bf - rf) / (mx - mn) + 2
        else:
            h = (rf - gf) / (mx - mn) + 4
        h *= 60
    s = 0 if mx == 0 else (mx - mn) / mx
    v = mx
    return h, s, v


def label_for(rgb: tuple[int, int, int], rank: int) -> str:
    """Heuristic role assignment from HSV characteristics."""
    _, s, v = rgb_to_hsv(*rgb)
    if s > 0.55 and v > 0.4:
        return "vibrant accent · CTA / badge"
    if v > 0.85 and s < 0.25:
        return "near-white · headline space"
    if v < 0.2:
        return "deep ink · text / contrast"
    if s < 0.2:
        return "muted neutral · background"
    return ROLE_LABELS[min(rank, len(ROLE_LABELS) - 1)]


def dominant_colors(img_path: Path, k: int = N_CLUSTERS) -> list[tuple[tuple[int, int, int], float]]:
    """Return [(rgb, score), …] sorted by saliency. Score = popularity × saturation
    so vivid accent colors out-rank the bland background majority.

    Filters more aggressively than v1:
      - drops near-white (V > 0.92, S < 0.1) and near-black (V < 0.06)
      - drops near-greys (S < 0.12 except very dark or very light)
      - clusters in HSV-rescaled space so similar hues group naturally
    """
    img = Image.open(img_path).convert("RGB").resize((SAMPLE_SIZE, SAMPLE_SIZE))
    pixels = np.asarray(img).reshape(-1, 3).astype(np.float32) / 255.0

    # Convert to HSV for filtering
    mx = pixels.max(axis=1)
    mn = pixels.min(axis=1)
    delta = mx - mn
    s_chan = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0)

    # Filter: drop near-greys, near-whites, near-blacks
    keep = (mx > 0.06) & (mx < 0.96)
    keep &= ~((s_chan < 0.12) & (mx > 0.12) & (mx < 0.92))
    if keep.sum() < 60:
        keep = np.ones(len(pixels), dtype=bool)
    pixels = pixels[keep]

    # Cluster in scaled-HSV-ish space: emphasize hue, downweight value
    rgb_for_kmeans = pixels.copy()
    km = KMeans(n_clusters=k, n_init=6, random_state=42).fit(rgb_for_kmeans)
    centers = km.cluster_centers_
    counts = np.bincount(km.labels_, minlength=k)

    out: list[tuple[tuple[int, int, int], float]] = []
    for i in range(k):
        r, g, b = (int(centers[i, 0] * 255), int(centers[i, 1] * 255), int(centers[i, 2] * 255))
        _, sat, val = rgb_to_hsv(r, g, b)
        # Score weights popularity, saturation, and a mild bonus for mid-brightness
        sat_boost = 0.45 + 0.55 * sat
        val_boost = 0.7 + 0.3 * (1.0 - abs(val - 0.55) * 2)  # peak around V=0.55
        score = float(counts[i]) * sat_boost * val_boost
        out.append(((r, g, b), score))
    out.sort(key=lambda t: -t[1])
    return out


def main() -> None:
    summary = pd.read_csv(DATA / "creative_summary.csv")
    print(f"Loaded {len(summary):,} creatives")

    # We want per-vertical palettes from real top performers.
    top = summary[summary["creative_status"] == "top_performer"]
    print(f"  top performers: {len(top):,}")

    per_vertical: dict[str, list[tuple[tuple[int, int, int], float]]] = {}

    for _, row in top.iterrows():
        cid = int(row["creative_id"])
        path = ASSETS / f"creative_{cid}.png"
        if not path.exists():
            continue
        try:
            colors = dominant_colors(path)
        except Exception:
            continue
        v = str(row["vertical"])
        per_vertical.setdefault(v, []).extend(colors)

    def aggregate(entries: list[tuple[tuple[int, int, int], float]], k_keep: int) -> list[dict]:
        """Bin by 4-bit-per-channel cells, score-weighted; pick the most salient
        distinct shades, then enforce hue diversity so we don't ship 5 shades
        of the same orange."""
        bucket: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]] = {}
        for rgb, sc in entries:
            key = (rgb[0] >> 4, rgb[1] >> 4, rgb[2] >> 4)
            bucket.setdefault(key, []).append((rgb, sc))
        scored: list[tuple[float, tuple[int, int, int]]] = []
        for members in bucket.values():
            arr = np.array([m[0] for m in members])
            mean_rgb = tuple(int(x) for x in arr.mean(axis=0))
            score_total = float(sum(m[1] for m in members))
            scored.append((score_total, mean_rgb))
        scored.sort(key=lambda t: -t[0])

        # Pick top-K with hue diversity: each new color must differ in either
        # RGB-distance OR hue-angle from everything already kept.
        kept_rgb: list[tuple[int, int, int]] = []
        kept_hue: list[float] = []
        out: list[dict] = []
        for _, rgb in scored:
            if any(np.linalg.norm(np.array(rgb) - np.array(k_)) < 28 for k_ in kept_rgb):
                continue
            h, s_, _ = rgb_to_hsv(*rgb)
            # If this color is colorful, also require a hue-distance gap of >25°
            if s_ > 0.18 and any(min(abs(h - kh), 360 - abs(h - kh)) < 25 for kh in kept_hue):
                continue
            kept_rgb.append(rgb)
            kept_hue.append(h)
            out.append({"hex": hex_of(rgb), "label": label_for(rgb, len(out))})
            if len(out) >= k_keep:
                break
        return out

    palettes_per_vertical = {v: aggregate(e, TOP_K_PER_VERTICAL) for v, e in per_vertical.items()}

    out_path = REPO / "front/public/data/palettes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "k-means dominant-color extraction over top-performer creatives",
        "k_clusters": N_CLUSTERS,
        "n_top_performers": int(top.shape[0]),
        "per_vertical": palettes_per_vertical,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO)}")
    for v, palette in palettes_per_vertical.items():
        print(f"  {v:18s} → {[c['hex'] for c in palette]}")


if __name__ == "__main__":
    main()
