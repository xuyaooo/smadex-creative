# 04 · Visual intelligence

> [← 03 · Tabular models](03_tabular_models.md) · [↑ Index](../README.md) · [05 · Personalized VLM + Flux edit →](05_personalized_vlm_and_image.md)

The visual side of the pipeline — embeddings, clusters, retrieval, the
data-grounded color palettes, and the LLM rubric features that augment
the tabular model.

These pieces are **optional for Model 1** (the tabular ensemble already
hits 0.677 macro-F1 without them) but **critical for the front-end's
Explorer and Predict pages**, and they are the inputs to Models 2 and 3.

## Visual encoder + embeddings

**Implementation**: [`scripts/precompute_embeddings.py`](../scripts/precompute_embeddings.py),
[`scripts/recompute_embeddings.py`](../scripts/recompute_embeddings.py),
[`src/embeddings/clip_encoder.py`](../src/embeddings/clip_encoder.py).

| Variant | Model | Shape | File | Size |
|---|---|---|---|---|
| Production | SigLIP-2 base | 768 → L2-norm + PCA(64) | `outputs/embeddings/clip_embeddings.npz` | 1.5 MB |
| Legacy | CLIP ViT-B/32 | 512 → PCA(32) | (overwritten by SigLIP-2) | — |

```bash
PYTHONPATH=models python3 models/scripts/precompute_embeddings.py
```

The `.npz` is keyed by `creative_id` so downstream code is shape-agnostic.

## UMAP + HDBSCAN clusters

UMAP (a non-linear projection that lays out high-dimensional embeddings
in 2-D while keeping nearby points nearby) gives the front-end its
scatter; HDBSCAN (density-based clustering — keeps a cluster only if
it's dense enough, otherwise marks it as noise) gives the cluster ids.

**Implementation**: [`scripts/build_artifacts.py`](../scripts/build_artifacts.py),
[`scripts/name_clusters.py`](../scripts/name_clusters.py).

| Artefact | Path | Shape |
|---|---|---|
| 2-D coords | `outputs/clusters/umap_coords.npz` | 1,076 × 2 |
| HDBSCAN labels | `outputs/clusters/labels.parquet` | `(creative_id, cluster_id, umap_x, umap_y)` |
| Deterministic names | `outputs/clusters/cluster_names.parquet` | `(cluster_id, name)` |

`name_clusters.py` walks each cluster, picks the modal `vertical · format ·
hook_type` triple, and writes a human-readable label like
*"gaming · interstitial · gameplay-tease"*. No LLM — fully deterministic.

## Per-vertical kNN

**Implementation**: [`scripts/build_artifacts.py`](../scripts/build_artifacts.py).

A separate `sklearn.NearestNeighbors` index per vertical (kNN, k-nearest-
neighbour retrieval — pulls the k creatives closest to the query in the
embedding space) so retrieval stays on-domain:

```
outputs/knn/index.pkl    6.4 MB    {vertical: (NearestNeighbors, ids)}
```

Used by the front-end's "find similar creatives" path with optional MMR
diversification (Maximum Marginal Relevance — picks results that are
both close to the query *and* different from each other, so you don't
get five near-duplicates).

## Per-vertical color palettes

**Implementation**: [`scripts/build_palette_lookup.py`](../scripts/build_palette_lookup.py).

For each of the 6 verticals we run k-means (k=5) over the dominant
colors of all **top-performer** creatives in that vertical and emit a
hex palette + a label per swatch. The result lives at
`front/public/data/palettes.json` and powers the suggested-palette card
on the Predict page.

```bash
PYTHONPATH=models python3 models/scripts/build_palette_lookup.py
```

## LLM rubric features (optional)

**Implementation**: [`scripts/extract_rubric.py`](../scripts/extract_rubric.py),
[`src/training/openrouter_rubric.py`](../src/training/openrouter_rubric.py),
[`scripts/benchmark_rubric.py`](../scripts/benchmark_rubric.py).

A teacher VLM scores each creative on **15 dimensions** (visual hierarchy,
brand visibility, hook strength, copy density, …) on a 0–10 anchored scale,
and the result becomes 15 extra numeric features for the tabular model.

```bash
export OPENROUTER_API_KEY=sk-or-...
PYTHONPATH=models python3 models/scripts/extract_rubric.py \
    --model google/gemini-2.5-flash --workers 64    # ~$0.10 · ~5 min
```

Output: `outputs/rubric/rubric_scores.parquet` (1,076 × 15).

`benchmark_rubric.py` evaluates candidate (model × prompt) combos against
a 12-creative ground-truth set so the choice of teacher is grounded.

## How the visual side feeds the rest

```
SigLIP-2 + PCA(64) ──┬──► UMAP/HDBSCAN ──► front-end Explorer
                     ├──► per-vertical kNN ──► "find similar"
                     └──► concat into tabular features (optional, +≤3pp F1)

LLM rubric (15 dims) ──► concat into tabular features (optional, +1–2pp F1)

Top-performer images per vertical ──► k-means(5) ──► palettes.json
```

## Design decisions

- **SigLIP-2 over CLIP.** SigLIP-2 is the single biggest visual upgrade
  available in this size range — better zero-shot, better
  compositionality. CLIP is kept as a legacy fallback only.

- **Per-vertical kNN, not global.** A "find similar" query in *gaming*
  shouldn't surface a fintech ad. Splitting the index by vertical
  keeps retrieval on-domain and makes each index small enough to
  fit comfortably in memory.

- **Deterministic cluster names.** A small LLM call would write nicer
  labels but we'd lose reproducibility — a re-run could relabel the
  same cluster differently. The modal-metadata recipe is boring on
  purpose.

- **Visual signal is optional in the tabular model.** It adds a few
  points of macro-F1 on top of the tabular baseline. We keep the
  ensemble runnable without it so the CPU path stays tractable, and
  fold the visual side in when it's available.

---

[← 03 · Tabular models](03_tabular_models.md) · [↑ Index](../README.md) · [05 · Personalized VLM + Flux edit →](05_personalized_vlm_and_image.md)
