"""
Unified Creative Intelligence Pipeline.
Combines tabular model, CLIP, fatigue detector, VLM, and recommender
into a single inference interface.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.data.feature_engineering import TabularFeatureEngineer
from src.data.loader import DataLoader
from src.embeddings.clip_encoder import CLIPCreativeEncoder, EmbeddingCache
from src.models.fatigue_detector import FatigueDetector
from src.models.recommender import CreativeRecommender
from src.models.tabular_model import XGBoostPerformancePredictor
from src.models.vlm_model import VLMCreativeAnalyzer

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

    def _ensure_data(self) -> None:
        if self._master_df is None:
            self._master_df = self.data_loader.load_master_table()
            self._daily_df = self.data_loader.load_daily_stats()
            X_tab, self._feature_names = self.feature_engineer.fit_transform(self._master_df)

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
