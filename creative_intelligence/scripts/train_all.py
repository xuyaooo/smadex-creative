"""Single-script training for tabular + fatigue models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, classification_report
import xgboost as xgb

from src.data.loader import DataLoader
from src.data.feature_engineering import TabularFeatureEngineer
from src.embeddings.clip_encoder import EmbeddingCache
from src.models.fatigue_detector import FatigueDetector

CONFIG = "config.yaml"

def train_tabular():
    print("=== Training Tabular Model ===")
    loader = DataLoader(CONFIG)
    df = loader.load_master_table()
    eng = TabularFeatureEngineer()
    X_tab, names = eng.fit_transform(df)
    y_perf = eng.get_perf_scores(df)
    y_status = eng.get_status_labels(df)
    groups = df["campaign_id"].values

    cache = EmbeddingCache("outputs/embeddings/clip_embeddings.npz")
    embeddings, ids = cache.load()
    id_to_emb = {cid: embeddings[i] for i, cid in enumerate(ids)}
    X_clip = np.stack([id_to_emb.get(int(c), np.zeros(512)) for c in df["creative_id"]])

    pca = PCA(n_components=32)
    X_clip_r = pca.fit_transform(X_clip)
    X = np.concatenate([X_tab, X_clip_r], axis=1)
    print(f"Feature matrix: {X.shape}")

    # CV eval
    gkf = GroupKFold(n_splits=5)
    val_maes = []
    for tr, va in gkf.split(X, y_perf, groups):
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                              tree_method="hist", random_state=42, verbosity=0)
        m.fit(X[tr], y_perf[tr])
        val_maes.append(mean_absolute_error(y_perf[va], m.predict(X[va])))
    print(f"CV Perf MAE: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")

    # Final fit
    perf_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   tree_method="hist", random_state=42, verbosity=0)
    perf_model.fit(X, y_perf)

    status_model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      objective="multi:softprob", num_class=4,
                                      tree_method="hist", random_state=42, verbosity=0)
    status_model.fit(X, y_status)

    preds = status_model.predict(X)
    print("Status train report:")
    print(classification_report(y_status, preds,
          target_names=["top_performer","stable","fatigued","underperformer"]))

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    perf_model.save_model("outputs/models/xgb_perf.json")
    status_model.save_model("outputs/models/xgb_status.json")
    with open("outputs/models/tabular_meta.pkl", "wb") as f:
        pickle.dump({"pca": pca,
                     "feature_names": names + [f"clip_pc{i}" for i in range(32)],
                     "_pca_fitted": True}, f)
    print("Tabular models saved.\n")

    # Feature importances top 10
    fi = sorted(zip(names + [f"clip_pc{i}" for i in range(32)],
                    perf_model.feature_importances_),
                key=lambda x: -x[1])[:10]
    print("Top 10 features (perf model):")
    for name, imp in fi:
        print(f"  {name}: {imp:.4f}")

def train_fatigue():
    print("\n=== Training Fatigue Detector ===")
    import yaml
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    loader = DataLoader(CONFIG)
    df = loader.load_master_table()
    daily = loader.load_daily_stats()
    print(f"Creatives: {len(df)}, Daily rows: {len(daily)}")

    detector = FatigueDetector(cfg["fatigue_model"])
    detector.fit(daily, df)
    detector.save(cfg["fatigue_model"])
    print("Fatigue detector saved.")

if __name__ == "__main__":
    train_tabular()
    train_fatigue()
    print("\nAll training complete!")
