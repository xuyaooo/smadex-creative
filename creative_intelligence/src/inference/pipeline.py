"""
Unified Creative Intelligence Pipeline.
Combines tabular model, CLIP, fatigue detector, VLM, and recommender
into a single inference interface.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import pickle

from src.calibration.temperature import TemperatureScaler
from src.data.early_features import compute_early_features
from src.data.feature_engineering import TabularFeatureEngineer
from src.data.loader import DataLoader
from src.data.rubric_features import align_rubric
from src.embeddings.clip_encoder import CLIPCreativeEncoder, EmbeddingCache
from src.fatigue.bocpd import fatigue_changepoint
from src.fatigue.health_score import health_score
from src.inference.explainer import counterfactual_suggestion, explain_creative
from src.models.fatigue_detector import FatigueDetector
from src.models.recommender import CreativeRecommender
from src.models.tabular_model import XGBoostPerformancePredictor
from src.models.vlm_model import VLMCreativeAnalyzer

RUBRIC_PARQUET = "outputs/rubric/rubric_scores.parquet"
TEMPERATURE_PATH = "outputs/models/temperature.pkl"
KNN_PATH = "outputs/knn/index.pkl"
CLUSTER_LABELS_PATH = "outputs/clusters/labels.parquet"
CLUSTER_NAMES_PATH = "outputs/clusters/cluster_names.parquet"

STATUS_LABELS = ["top_performer", "stable", "fatigued", "underperformer"]


@dataclass
class CreativeReport:
    creative_id: int
    perf_score: float
    predicted_status: str
    status_probabilities: Dict[str, float]
    shap_top_features: Dict[str, float]
    fatigue_risk: Dict
    vlm_analysis: Dict
    recommendations: List[Dict]
    creative_brief: Dict


class CreativeIntelligencePipeline:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self._cfg = cfg
        self._config_path = config_path
        self._root = Path(config_path).parent

        self.data_loader = DataLoader(config_path)
        self.feature_engineer = TabularFeatureEngineer()
        # CLIP encoder is lazy — only initialized when encoding a new image.
        # When the embedding cache exists (precomputed), CLIP is never loaded,
        # avoiding an OpenMP conflict between PyTorch and XGBoost on macOS.
        self._clip_encoder: Optional[CLIPCreativeEncoder] = None
        self.embedding_cache = EmbeddingCache(
            self._root / cfg["embeddings"]["cache_path"]
        )

        self._master_df: Optional[pd.DataFrame] = None
        self._daily_df: Optional[pd.DataFrame] = None
        self._tabular_model: Optional[XGBoostPerformancePredictor] = None
        self._fatigue_detector: Optional[FatigueDetector] = None
        self._vlm_analyzer: Optional[VLMCreativeAnalyzer] = None
        self._recommender: Optional[CreativeRecommender] = None
        self._feature_names: List[str] = []
        self._early_by_cid: Dict[int, np.ndarray] = {}
        self._n_early_features: int = 0
        self._rubric_by_cid: Dict[int, np.ndarray] = {}
        self._n_rubric_features: int = 0
        self._temperature: Optional[TemperatureScaler] = None
        self._perf_quantiles_by_vertical: Dict[str, np.ndarray] = {}
        self._knn_by_vertical: Dict = {}
        self._cluster_by_cid: Dict[int, int] = {}
        self._cluster_names: Dict[int, str] = {}
        self._umap_by_cid: Dict[int, Tuple[float, float]] = {}
        self._rubric_names: List[str] = []
        self._rubric_importance: Dict[str, float] = {}

    def _ensure_data(self) -> None:
        if self._master_df is None:
            self._master_df = self.data_loader.load_master_table()
            self._daily_df = self.data_loader.load_daily_stats()
            X_tab, self._feature_names = self.feature_engineer.fit_transform(self._master_df)

            cids = self._master_df["creative_id"].astype(int).tolist()
            early_X, _ = compute_early_features(self._daily_df, cids, window=7)
            self._early_by_cid = {cid: early_X[i] for i, cid in enumerate(cids)}
            self._n_early_features = early_X.shape[1]

            rubric_X, _ = align_rubric(self._root / RUBRIC_PARQUET, cids)
            self._rubric_by_cid = {cid: rubric_X[i] for i, cid in enumerate(cids)}
            self._n_rubric_features = rubric_X.shape[1]

        if not self.embedding_cache.exists():
            self._precompute_embeddings()

    @property
    def clip_encoder(self) -> CLIPCreativeEncoder:
        if self._clip_encoder is None:
            self._clip_encoder = CLIPCreativeEncoder(
                model_name=self._cfg["embeddings"]["clip_model"]
            )
        return self._clip_encoder

    def _precompute_embeddings(self) -> None:
        paths = [
            self.data_loader.get_asset_path(int(cid))
            for cid in self._master_df["creative_id"]
        ]
        valid = [(cid, p) for cid, p in zip(self._master_df["creative_id"], paths) if p.exists()]
        ids = [int(c) for c, _ in valid]
        image_paths = [str(p) for _, p in valid]
        embeddings = self.clip_encoder.encode_batch(
            image_paths, batch_size=self._cfg["embeddings"]["batch_size"]
        )
        self.embedding_cache.save(embeddings, ids)

    def _ensure_models(self) -> None:
        self._ensure_data()
        tab_cfg = self._cfg["tabular_model"]
        fat_cfg = self._cfg["fatigue_model"]
        vlm_cfg = self._cfg["vlm"]
        vlm_cfg["device"] = self._cfg["inference"]["device"]

        if self._tabular_model is None:
            self._tabular_model = XGBoostPerformancePredictor.load(tab_cfg)

        if self._fatigue_detector is None:
            self._fatigue_detector = FatigueDetector.load(fat_cfg)

        if self._temperature is None:
            tpath = self._root / TEMPERATURE_PATH
            self._temperature = TemperatureScaler.load(tpath) if tpath.exists() else TemperatureScaler()

        if not self._perf_quantiles_by_vertical and "vertical" in self._master_df.columns:
            for v, sub in self._master_df.groupby("vertical"):
                self._perf_quantiles_by_vertical[v] = np.sort(sub["perf_score"].fillna(0).values)

        # KNN, clusters, cluster names — all precomputed parquets / pickle.
        knn_path = self._root / KNN_PATH
        if not self._knn_by_vertical and knn_path.exists():
            with open(knn_path, "rb") as f:
                self._knn_by_vertical = pickle.load(f)

        labels_path = self._root / CLUSTER_LABELS_PATH
        if not self._cluster_by_cid and labels_path.exists():
            cdf = pd.read_parquet(labels_path)
            for r in cdf.itertuples(index=False):
                self._cluster_by_cid[int(r.creative_id)] = int(r.cluster_id)
                self._umap_by_cid[int(r.creative_id)] = (float(r.umap_x), float(r.umap_y))

        names_path = self._root / CLUSTER_NAMES_PATH
        if not self._cluster_names and names_path.exists():
            ndf = pd.read_parquet(names_path)
            self._cluster_names = {int(r.cluster_id): r.name for r in ndf.itertuples(index=False)}

        # Rubric importances — used by counterfactual_suggestion. Read from XGBoost.
        if not self._rubric_importance and self._tabular_model is not None:
            try:
                booster = self._tabular_model.perf_model.get_booster()
                imp = booster.get_score(importance_type="gain")
                fnames = self._tabular_model.feature_names
                self._rubric_importance = {
                    fnames[int(k.removeprefix("f"))]: float(v) / sum(imp.values())
                    for k, v in imp.items()
                    if int(k.removeprefix("f")) < len(fnames)
                }
            except Exception:
                self._rubric_importance = {}

        if self._vlm_analyzer is None:
            vlm_ckpt = self._root / vlm_cfg.get("student_checkpoint", "outputs/models/vlm_finetuned")
            if vlm_ckpt.exists():
                self._vlm_analyzer = VLMCreativeAnalyzer.load(vlm_cfg)

        if self._recommender is None:
            self._recommender = CreativeRecommender(
                master_df=self._master_df.reset_index(drop=True),
                embedding_cache=self.embedding_cache,
                tabular_model=self._tabular_model,
                top_k=self._cfg["inference"]["top_k_similar"],
            )

    def analyze_creative(self, creative_id: int) -> CreativeReport:
        self._ensure_models()

        row = self._master_df[self._master_df["creative_id"] == creative_id]
        if row.empty:
            raise ValueError(f"Creative {creative_id} not found in dataset.")

        row = row.iloc[0]
        X_tab = self.feature_engineer.transform(pd.DataFrame([row]))
        early = self._early_by_cid.get(int(creative_id), np.zeros(self._n_early_features, dtype=np.float32))
        rubric = self._rubric_by_cid.get(int(creative_id), np.zeros(self._n_rubric_features, dtype=np.float32))
        X_tab = np.concatenate([X_tab, early.reshape(1, -1), rubric.reshape(1, -1)], axis=1)
        clip_emb = self.embedding_cache.get_embedding(creative_id)
        if clip_emb is None:
            image_path = self.data_loader.get_asset_path(creative_id)
            clip_emb = self.clip_encoder.encode_image(str(image_path))

        X_clip = clip_emb.reshape(1, -1)

        perf_score = float(self._tabular_model.predict_perf_score(X_tab, X_clip)[0])
        status_labels, status_probs = self._tabular_model.predict_status(X_tab, X_clip)
        predicted_status = STATUS_LABELS[int(status_labels[0])]
        status_probabilities = dict(zip(STATUS_LABELS, status_probs[0].tolist()))

        shap_values = self._tabular_model.explain_prediction(X_tab, X_clip)
        shap_top = dict(sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        fatigue_risk = self._fatigue_detector.predict_fatigue_risk(creative_id, self._daily_df)

        image_path = self.data_loader.get_asset_path(creative_id)
        vlm_analysis: Dict = {}
        if self._vlm_analyzer is not None and image_path.exists():
            vlm_analysis = self._vlm_analyzer.analyze(str(image_path), row.to_dict())

        recommendations = self._recommender.recommend_feature_changes(creative_id, shap_values)
        similar = self._recommender.retrieve_similar_top_performers(creative_id)
        brief = self._recommender.generate_creative_brief(creative_id, shap_values, vlm_analysis)

        return CreativeReport(
            creative_id=creative_id,
            perf_score=perf_score,
            predicted_status=predicted_status,
            status_probabilities=status_probabilities,
            shap_top_features=shap_top,
            fatigue_risk=fatigue_risk,
            vlm_analysis=vlm_analysis,
            recommendations=recommendations,
            creative_brief=brief,
        )

    def find_similar(
        self,
        creative_id: int,
        k: int = 5,
        scope: str = "vertical",
        diversify: bool = False,
    ) -> List[Dict]:
        """Vertical-scoped or global nearest-neighbor lookup. Returns top-k similar
        creatives with cosine similarity. <5 ms.

        When ``diversify=True``, pulls a wider candidate pool (~30) from FAISS,
        re-ranks by ``0.7 * sim + 0.3 * perf_score`` to surface high-performing
        peers, then runs greedy MMR (lambda=0.5) over the candidate embeddings
        to return a *diverse* slate of k creatives instead of k near-duplicates.
        """
        self._ensure_models()
        row = self._master_df[self._master_df["creative_id"] == creative_id]
        if row.empty:
            return []
        vertical = row.iloc[0].get("vertical", "_all_")
        index = self._knn_by_vertical.get(vertical) if scope == "vertical" else None
        index = index or self._knn_by_vertical.get("_all_")
        if index is None:
            return []

        emb = self.embedding_cache.get_embedding(creative_id)
        if emb is None:
            return []
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        # In diversify mode pull a wider pool for re-rank + MMR; otherwise k+1
        # (one slot is the query creative itself).
        pool = 30 if diversify else (k + 1)
        n_request = min(pool, len(index["cids"]))
        dists, idxs = index["nn"].kneighbors(emb.reshape(1, -1), n_neighbors=n_request)

        # Build the candidate set (drop the query itself).
        cand_cids: List[int] = []
        cand_sims: List[float] = []
        cand_perfs: List[float] = []
        cand_status: List[str] = []
        cand_local_idx: List[int] = []  # row in index["embeddings_normed"]
        for d, i in zip(dists[0], idxs[0]):
            cid = int(index["cids"][int(i)])
            if cid == int(creative_id):
                continue
            sub = self._master_df[self._master_df["creative_id"] == cid]
            status = sub.iloc[0]["creative_status"] if not sub.empty else "?"
            perf = float(sub.iloc[0]["perf_score"]) if not sub.empty else 0.0
            cand_cids.append(cid)
            cand_sims.append(1.0 - float(d))
            cand_perfs.append(perf)
            cand_status.append(status)
            cand_local_idx.append(int(i))

        if not diversify:
            return [
                {
                    "creative_id": cid,
                    "similarity": round(sim, 4),
                    "creative_status": status,
                    "perf_score": round(perf, 4),
                }
                for cid, sim, status, perf in zip(
                    cand_cids[:k], cand_sims[:k], cand_status[:k], cand_perfs[:k]
                )
            ]

        # ----- diversify=True: rerank by perf, then MMR over embeddings -----
        from src.inference.dpp_recommender import mmr_diversify, rerank_by_perf

        positions = list(range(len(cand_cids)))
        # Re-rank gives us an ordered position list; turn it into a relevance
        # vector aligned to original positions for MMR (higher rank → higher rel).
        ranked = rerank_by_perf(positions, cand_sims, cand_perfs, alpha=0.7)
        rel = np.zeros(len(positions), dtype=np.float64)
        for rank, pos in enumerate(ranked):
            rel[pos] = len(positions) - rank

        embeddings_normed = index.get("embeddings_normed")
        if embeddings_normed is None or not cand_local_idx:
            # Fallback: no per-vertical embedding matrix → just use rerank order.
            picks = ranked[:k]
        else:
            cand_embs = np.asarray(embeddings_normed)[cand_local_idx]
            picks = mmr_diversify(cand_embs, rel, k=k, lambda_=0.5)

        return [
            {
                "creative_id": cand_cids[p],
                "similarity": round(cand_sims[p], 4),
                "creative_status": cand_status[p],
                "perf_score": round(cand_perfs[p], 4),
            }
            for p in picks
        ]

    def cluster_info(self, creative_id: int) -> Dict:
        """Cluster ID + human name + 5 nearby cluster members. <2 ms."""
        self._ensure_models()
        cid_int = int(creative_id)
        cluster_id = self._cluster_by_cid.get(cid_int, -1)
        name = self._cluster_names.get(cluster_id, "Unknown")
        coords = self._umap_by_cid.get(cid_int, (0.0, 0.0))

        members = [c for c, cl in self._cluster_by_cid.items() if cl == cluster_id and c != cid_int][:8]
        return {
            "creative_id": cid_int,
            "cluster_id": cluster_id,
            "cluster_name": name,
            "umap_xy": coords,
            "n_members": sum(1 for cl in self._cluster_by_cid.values() if cl == cluster_id),
            "sample_members": members,
        }

    def explain(self, creative_id: int) -> Dict:
        """One-shot explanation: SHAP top features + rubric callouts + counterfactual. <80 ms."""
        self._ensure_models()
        row = self._master_df[self._master_df["creative_id"] == creative_id]
        if row.empty:
            raise ValueError(f"Creative {creative_id} not found.")
        row = row.iloc[0]

        X_tab = self.feature_engineer.transform(pd.DataFrame([row]))
        early = self._early_by_cid.get(int(creative_id), np.zeros(self._n_early_features, dtype=np.float32))
        rubric_vec = self._rubric_by_cid.get(int(creative_id), np.zeros(self._n_rubric_features, dtype=np.float32))
        X_tab = np.concatenate([X_tab, early.reshape(1, -1), rubric_vec.reshape(1, -1)], axis=1)
        clip_emb = self.embedding_cache.get_embedding(creative_id)
        X_clip = clip_emb.reshape(1, -1)

        perf_pred = float(self._tabular_model.predict_perf_score(X_tab, X_clip)[0])
        shap = self._tabular_model.explain_prediction(X_tab, X_clip)

        vertical = row.get("vertical", "unknown")
        q = self._perf_quantiles_by_vertical.get(vertical, np.array([perf_pred]))
        percentile = float(np.searchsorted(q, perf_pred) / max(len(q), 1))

        # Rubric dict
        rubric_cols = [n for n in self._tabular_model.feature_names
                       if n in {"hook_clarity", "cta_prominence", "cta_contrast",
                                "color_vibrancy", "color_warmth", "text_density_visual",
                                "face_count_visual", "product_focus", "scene_realism",
                                "emotion_intensity", "composition_balance", "brand_visibility",
                                "urgency_signal", "playfulness", "novelty_visual"}]
        rubric_dict = {}
        for col in rubric_cols:
            try:
                idx = self._tabular_model.feature_names.index(col)
                if idx < X_tab.shape[1]:
                    rubric_dict[col] = int(X_tab[0, idx])
            except (ValueError, IndexError):
                pass

        h = self.health_score(creative_id)
        explanation = explain_creative(
            perf_pred=perf_pred,
            perf_percentile_vertical=percentile,
            vertical=str(vertical),
            shap_dict=shap,
            rubric=rubric_dict,
            health=h,
        )
        explanation["counterfactuals"] = counterfactual_suggestion(
            rubric_dict, perf_pred, self._rubric_importance, n_top=3
        )
        explanation["health"] = h
        return explanation

    def health_score(self, creative_id: int) -> Dict:
        """Single 0–100 health number with action recommendation. <100 ms."""
        self._ensure_models()
        row = self._master_df[self._master_df["creative_id"] == creative_id]
        if row.empty:
            raise ValueError(f"Creative {creative_id} not found.")
        row = row.iloc[0]

        # Predicted perf
        X_tab = self.feature_engineer.transform(pd.DataFrame([row]))
        early = self._early_by_cid.get(int(creative_id), np.zeros(self._n_early_features, dtype=np.float32))
        rubric = self._rubric_by_cid.get(int(creative_id), np.zeros(self._n_rubric_features, dtype=np.float32))
        X_tab = np.concatenate([X_tab, early.reshape(1, -1), rubric.reshape(1, -1)], axis=1)
        clip_emb = self.embedding_cache.get_embedding(creative_id)
        X_clip = clip_emb.reshape(1, -1)
        perf_pred = float(self._tabular_model.predict_perf_score(X_tab, X_clip)[0])

        # Calibrated 4-way status probs — used to override action selection
        # at the margins where perf_pred is conservative (esp. top_performer).
        _, raw_status_probs = self._tabular_model.predict_status(X_tab, X_clip)
        cal_probs = self._temperature.transform(raw_status_probs)[0]
        status_probs = dict(zip(STATUS_LABELS, cal_probs.tolist()))

        # Percentile within vertical
        vertical = row.get("vertical", "unknown")
        q = self._perf_quantiles_by_vertical.get(vertical, np.array([perf_pred]))
        percentile = float(np.searchsorted(q, perf_pred) / max(len(q), 1))

        # Fatigue probability + days remaining
        fatigue = self._fatigue_detector.predict_fatigue_risk(creative_id, self._daily_df)

        # BOCPD on the daily CTR series for this creative
        ts = self._daily_df[self._daily_df["creative_id"] == creative_id].sort_values("days_since_launch")
        agg = ts.groupby("days_since_launch").agg(impressions=("impressions", "sum"),
                                                  clicks=("clicks", "sum")).reset_index()
        ctr_series = (agg["clicks"] / agg["impressions"].replace(0, np.nan)).fillna(0).values
        cp = fatigue_changepoint(ctr_series, hazard_lambda=50.0, threshold=0.15)

        return health_score(
            perf_pred=perf_pred,
            perf_percentile_vertical=percentile,
            fatigue_prob=float(fatigue["fatigue_probability"]),
            has_changepoint=cp["has_changepoint"],
            days_active=int(fatigue["days_active"]),
            days_remaining_estimate=int(fatigue["days_remaining_estimate"]),
            status_probs=status_probs,
        ) | {
            "creative_id": int(creative_id),
            "vertical": vertical,
            "predicted_perf": round(perf_pred, 4),
            "changepoint": cp,
        }

    def monitor_campaign(self, campaign_id: int) -> pd.DataFrame:
        self._ensure_models()
        creatives = self._master_df[self._master_df["campaign_id"] == campaign_id]
        rows = []
        for _, row in creatives.iterrows():
            cid = int(row["creative_id"])
            fatigue = self._fatigue_detector.predict_fatigue_risk(cid, self._daily_df)
            rows.append({
                "creative_id": cid,
                "creative_status": row["creative_status"],
                "overall_ctr": row.get("overall_ctr", 0),
                "overall_ipm": row.get("overall_ipm", 0),
                "fatigue_probability": fatigue["fatigue_probability"],
                "current_fatigue_score": fatigue["current_fatigue_score"],
                "days_active": fatigue["days_active"],
            })
        return pd.DataFrame(rows).sort_values("fatigue_probability", ascending=False)

    def score_new_creative(self, image_path: str, metadata: Dict) -> Dict:
        self._ensure_models()
        X_tab = self.feature_engineer.transform(pd.DataFrame([metadata]))
        # Brand-new creative has no daily history and no rubric extracted yet.
        # Both feature blocks are zero-padded; downstream caller can score the rubric
        # offline later and re-call.
        X_tab = np.concatenate(
            [X_tab,
             np.zeros((1, self._n_early_features), dtype=np.float32),
             np.zeros((1, self._n_rubric_features), dtype=np.float32)],
            axis=1,
        )
        clip_emb = self.clip_encoder.encode_image(image_path).reshape(1, -1)
        perf_score = float(self._tabular_model.predict_perf_score(X_tab, clip_emb)[0])
        _, probs = self._tabular_model.predict_status(X_tab, clip_emb)

        vlm_analysis: Dict = {}
        if self._vlm_analyzer is not None:
            vlm_analysis = self._vlm_analyzer.analyze(image_path, metadata)

        return {
            "perf_score": perf_score,
            "predicted_status": STATUS_LABELS[probs[0].argmax()],
            "status_probabilities": dict(zip(STATUS_LABELS, probs[0].tolist())),
            "vlm_analysis": vlm_analysis,
        }
