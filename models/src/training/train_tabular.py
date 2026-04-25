"""Train the XGBoost tabular + CLIP model."""

from pathlib import Path

import numpy as np
import yaml

from src.data.loader import DataLoader
from src.data.feature_engineering import TabularFeatureEngineer
from src.embeddings.clip_encoder import EmbeddingCache
from src.models.tabular_model import XGBoostPerformancePredictor


def train(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(config_path).parent

    print("Loading data...")
    loader = DataLoader(config_path)
    master_df = loader.load_master_table()

    print("Engineering features...")
    engineer = TabularFeatureEngineer()
    X_tab, feature_names = engineer.fit_transform(master_df)
    y_perf = engineer.get_perf_scores(master_df)
    y_status = engineer.get_status_labels(master_df)
    groups = master_df["campaign_id"].values

    print("Loading CLIP embeddings...")
    cache = EmbeddingCache(root / cfg["embeddings"]["cache_path"])
    if not cache.exists():
        raise FileNotFoundError(
            "CLIP embeddings not found. Run: python scripts/precompute_embeddings.py"
        )
    embeddings, ids = cache.load()

    # Align embeddings with master_df order
    id_to_emb = {cid: embeddings[i] for i, cid in enumerate(ids)}
    X_clip = np.stack([
        id_to_emb.get(int(cid), np.zeros(embeddings.shape[1]))
        for cid in master_df["creative_id"]
    ])

    print("Training models...")
    tab_cfg = cfg["tabular_model"]
    tab_cfg["pca_components"] = cfg["embeddings"]["pca_components"]
    model = XGBoostPerformancePredictor(tab_cfg)
    metrics = model.fit(X_tab, X_clip, y_perf, y_status, groups, feature_names)
    print(f"CV results: {metrics}")

    model.save(tab_cfg)
    print("Tabular model saved.")

    # Print top feature importances
    fi = model.get_feature_importances().head(15)
    print("\nTop 15 features:")
    print(fi.to_string(index=False))


if __name__ == "__main__":
    train()
