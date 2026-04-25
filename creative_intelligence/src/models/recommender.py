from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.embeddings.clip_encoder import EmbeddingCache


STATUS_PRIORITY = {"top_performer": 0, "stable": 1, "underperformer": 2, "fatigued": 3}


class CreativeRecommender:
    def __init__(
        self,
        master_df: pd.DataFrame,
        embedding_cache: EmbeddingCache,
        tabular_model=None,
        top_k: int = 5,
    ):
        self.master_df = master_df.set_index("creative_id")
        self.embedding_cache = embedding_cache
        self.tabular_model = tabular_model
        self.top_k = top_k
        self._embeddings, self._ids = embedding_cache.get_all()

    def _get_embedding(self, creative_id: int) -> np.ndarray:
        emb = self.embedding_cache.get_embedding(creative_id)
        return emb if emb is not None else np.zeros(self._embeddings.shape[1])

    def retrieve_similar_top_performers(
        self, creative_id: int, same_vertical: bool = True
    ) -> List[Dict]:
        query_emb = self._get_embedding(creative_id)
        query_row = self.master_df.loc[creative_id] if creative_id in self.master_df.index else {}

        # Filter to good performers in the same vertical
        candidates = self.master_df[
            self.master_df["creative_status"].isin(["top_performer", "stable"])
        ].copy()

        if same_vertical and hasattr(query_row, "get") and query_row.get("vertical"):
            vertical_mask = candidates["vertical"] == query_row.get("vertical")
            if vertical_mask.sum() >= self.top_k:
                candidates = candidates[vertical_mask]

        # Cosine similarity
        candidate_ids = [cid for cid in candidates.index if cid in self._ids]
        if not candidate_ids:
            return []

        idx_map = {cid: self._ids.index(cid) for cid in candidate_ids}
        cand_embs = np.stack([self._embeddings[idx_map[cid]] for cid in candidate_ids])
        sims = cand_embs @ query_emb / (
            np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        top_indices = np.argsort(-sims)[: self.top_k]

        results = []
        for i in top_indices:
            cid = candidate_ids[i]
            row = candidates.loc[cid]
            results.append({
                "creative_id": int(cid),
                "similarity": float(sims[i]),
                "status": str(row.get("creative_status", "unknown")),
                "ctr": float(row.get("overall_ctr", 0)),
                "ipm": float(row.get("overall_ipm", 0)),
                "format": str(row.get("format", "unknown")),
                "theme": str(row.get("theme", "unknown")),
                "dominant_color": str(row.get("dominant_color", "unknown")),
            })
        return results

    def recommend_feature_changes(self, creative_id: int, shap_values: Dict) -> List[Dict]:
        if creative_id not in self.master_df.index:
            return []
        query_row = self.master_df.loc[creative_id]
        vertical = query_row.get("vertical", None)

        top_performers = self.master_df[
            (self.master_df["creative_status"] == "top_performer") &
            (self.master_df["vertical"] == vertical if vertical else True)
        ]

        # Sort SHAP values to find top dragging features (most negative)
        negative_features = sorted(
            [(feat, val) for feat, val in shap_values.items() if val < 0],
            key=lambda x: x[1]
        )[:5]

        recs = []
        numeric_cols = ["clutter_score", "novelty_score", "readability_score",
                        "brand_visibility_score", "motion_score", "text_density"]

        for feat, shap_val in negative_features:
            if feat not in numeric_cols or feat not in top_performers.columns:
                continue
            current_val = float(query_row.get(feat, 0))
            tp_range = top_performers[feat].dropna()
            if tp_range.empty:
                continue
            recommended_range = f"{tp_range.quantile(0.25):.2f}–{tp_range.quantile(0.75):.2f}"
            recs.append({
                "feature": feat,
                "current_value": round(current_val, 3),
                "top_performer_range": recommended_range,
                "shap_impact": round(shap_val, 4),
                "rationale": (
                    f"Top performers in {vertical} have {feat} in {recommended_range}; "
                    f"yours is {current_val:.2f}"
                ),
            })
        return recs

    def generate_creative_brief(
        self,
        creative_id: int,
        shap_values: Dict,
        vlm_analysis: Dict,
    ) -> Dict:
        similar = self.retrieve_similar_top_performers(creative_id)
        changes = self.recommend_feature_changes(creative_id, shap_values)
        row = self.master_df.loc[creative_id] if creative_id in self.master_df.index else {}

        keep = []
        if str(row.get("format", "")) in ["rewarded_video", "native"]:
            keep.append(f"Keep {row.get('format')} format (strong in {row.get('vertical')})")
        if str(row.get("dominant_color", "")) not in ["", "unknown"]:
            keep.append(f"Keep {row.get('dominant_color')} color palette if brand-consistent")

        change = [c["rationale"] for c in changes[:3]]
        top_rec = vlm_analysis.get("top_recommendation", "Reduce clutter and increase CTA clarity")

        test_variations = []
        if similar:
            for s in similar[:2]:
                test_variations.append({
                    "based_on_creative": s["creative_id"],
                    "theme": s.get("theme", "?"),
                    "format": s.get("format", "?"),
                    "similarity_score": round(s["similarity"], 3),
                })

        return {
            "brief_for": f"new creative to replace {creative_id}",
            "keep": keep,
            "change": change,
            "top_recommendation": top_rec,
            "test_variations": test_variations,
            "vlm_rationale": vlm_analysis.get("performance_summary", ""),
            "visual_weaknesses": vlm_analysis.get("visual_weaknesses", []),
        }
