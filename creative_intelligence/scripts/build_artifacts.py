"""
P4 — Precompute every artifact the demo needs at runtime.

After this runs, the Gradio demo has zero heavy work to do at query time:
  - knn/index.pkl        → vertical-scoped NearestNeighbors over visual embeddings
  - clusters/labels.parquet → UMAP coords + HDBSCAN cluster_id per creative
  - shap/background.npz  → SHAP background sample (small) for fast TreeSHAP
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

from src.data.loader import DataLoader
from src.embeddings.clip_encoder import EmbeddingCache

CONFIG = "config.yaml"

OUT_KNN = Path("outputs/knn/index.pkl")
OUT_CLUSTERS = Path("outputs/clusters/labels.parquet")
OUT_SHAP_BG = Path("outputs/shap/background.npz")
OUT_UMAP = Path("outputs/clusters/umap_coords.npz")


def build_knn(master: pd.DataFrame, embeddings: np.ndarray, ids: list) -> None:
    """Per-vertical NearestNeighbors so 'find similar' is scoped to the same domain."""
    OUT_KNN.parent.mkdir(parents=True, exist_ok=True)
    cid_to_emb = {int(c): embeddings[i] for i, c in enumerate(ids)}

    by_vertical: dict = {}
    for vertical, sub in master.groupby("vertical"):
        cids = sub["creative_id"].astype(int).tolist()
        E = np.stack([cid_to_emb.get(c, np.zeros(embeddings.shape[1])) for c in cids])
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        nn = NearestNeighbors(n_neighbors=min(10, len(cids)), metric="cosine")
        nn.fit(E)
        by_vertical[str(vertical)] = {"nn": nn, "cids": cids, "embeddings_normed": E}

    # Global index (all verticals) for fallback / cold-start
    E_all = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    nn_all = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn_all.fit(E_all)
    by_vertical["_all_"] = {"nn": nn_all, "cids": [int(c) for c in ids], "embeddings_normed": E_all}

    with open(OUT_KNN, "wb") as f:
        pickle.dump(by_vertical, f)
    print(f"  knn: {len(by_vertical) - 1} verticals + global index → {OUT_KNN}")


def build_clusters(master: pd.DataFrame, embeddings: np.ndarray, ids: list) -> None:
    """UMAP-50d → HDBSCAN clusters; also a UMAP-2d coord pair for visualization."""
    import umap
    import hdbscan

    OUT_CLUSTERS.parent.mkdir(parents=True, exist_ok=True)
    cid_to_emb = {int(c): embeddings[i] for i, c in enumerate(ids)}

    cids = master["creative_id"].astype(int).tolist()
    E = np.stack([cid_to_emb.get(c, np.zeros(embeddings.shape[1])) for c in cids])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    print("  UMAP → 2d (for plotting)...")
    umap_2d = umap.UMAP(n_components=2, n_neighbors=15, metric="cosine", random_state=42)
    coords_2d = umap_2d.fit_transform(E)

    print("  UMAP → 30d (for clustering)...")
    umap_30 = umap.UMAP(n_components=30, n_neighbors=15, metric="cosine", random_state=42)
    coords_30 = umap_30.fit_transform(E)

    print("  HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean")
    cluster_ids = clusterer.fit_predict(coords_30)
    n_clusters = int(cluster_ids.max() + 1)
    n_noise = int((cluster_ids == -1).sum())
    print(f"  → {n_clusters} clusters, {n_noise} noise points (of {len(cids)})")

    df = pd.DataFrame({
        "creative_id": cids,
        "cluster_id": cluster_ids,
        "umap_x": coords_2d[:, 0],
        "umap_y": coords_2d[:, 1],
    })
    df.to_parquet(OUT_CLUSTERS, index=False)
    np.savez_compressed(OUT_UMAP, coords_2d=coords_2d, cluster_ids=cluster_ids,
                        creative_ids=np.array(cids))
    print(f"  clusters: {OUT_CLUSTERS}")


def build_shap_background(master: pd.DataFrame, embeddings: np.ndarray, ids: list) -> None:
    """Tiny stratified sample of training rows to use as SHAP background."""
    from src.data.feature_engineering import TabularFeatureEngineer
    from src.data.early_features import compute_early_features
    from src.data.rubric_features import align_rubric

    OUT_SHAP_BG.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(CONFIG)
    daily = loader.load_daily_stats()

    fe = TabularFeatureEngineer()
    X_tab, _ = fe.fit_transform(master)
    cids = master["creative_id"].astype(int).tolist()
    X_early, _ = compute_early_features(daily, cids, window=7)
    X_rubric, _ = align_rubric("outputs/rubric/rubric_scores.parquet", cids)

    cid_to_emb = {int(c): embeddings[i] for i, c in enumerate(ids)}
    X_clip = np.stack([cid_to_emb.get(c, np.zeros(embeddings.shape[1])) for c in cids])

    # Stratified sample: 8 from each true status bucket
    sample_idx = []
    for s in master["creative_status"].unique():
        mask = master["creative_status"] == s
        cand = np.where(mask.values)[0]
        rng = np.random.default_rng(42)
        sample_idx.extend(rng.choice(cand, size=min(8, len(cand)), replace=False).tolist())

    np.savez_compressed(
        OUT_SHAP_BG,
        X_tab=X_tab[sample_idx], X_early=X_early[sample_idx],
        X_rubric=X_rubric[sample_idx], X_clip=X_clip[sample_idx],
        creative_ids=np.array(cids)[sample_idx],
    )
    print(f"  shap background: {len(sample_idx)} rows → {OUT_SHAP_BG}")


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    loader = DataLoader(CONFIG)
    master = loader.load_master_table()
    cache = EmbeddingCache(cfg["embeddings"]["cache_path"])
    emb, ids = cache.get_all()
    print(f"master: {master.shape} | embeddings: {emb.shape}")

    t0 = time.time()
    print("\n[1/3] KNN indices (per-vertical)")
    build_knn(master, emb, ids)
    print("\n[2/3] Clusters (UMAP + HDBSCAN)")
    build_clusters(master, emb, ids)
    print("\n[3/3] SHAP background sample")
    build_shap_background(master, emb, ids)
    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
