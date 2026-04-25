"""End-to-end smoke test for CreativeIntelligencePipeline.

Marked as slow because cold-start loads XGBoost, LightGBM, KNN, clusters,
and a CLIP embedding cache. Skipped automatically if any precomputed
artifact is missing so the suite remains green on a clean clone.
"""
import os
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_ARTIFACTS = [
    PROJECT_ROOT / "outputs" / "models" / "xgb_perf.json",
    PROJECT_ROOT / "outputs" / "models" / "xgb_status.json",
    PROJECT_ROOT / "outputs" / "models" / "fatigue_clf.pkl",
    PROJECT_ROOT / "outputs" / "embeddings" / "clip_embeddings.npz",
]


pytestmark = pytest.mark.slow


def _missing_artifacts() -> list[str]:
    return [str(p) for p in REQUIRED_ARTIFACTS if not p.exists()]


@pytest.fixture(scope="module")
def pipeline():
    missing = _missing_artifacts()
    if missing:
        pytest.skip(f"missing precomputed artifacts: {missing}")
    # Pipeline reads relative paths from config -> chdir to project root
    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        from src.inference.pipeline import CreativeIntelligencePipeline
        p = CreativeIntelligencePipeline(str(CONFIG_PATH.name))
        t0 = time.time()
        p._ensure_models()
        cold_start = time.time() - t0
        # Sanity bound (5s requested by spec). Soft assertion lives below as a real test.
        p._cold_start_seconds = cold_start
        yield p
    finally:
        os.chdir(cwd)


def test_cold_start_is_fast(pipeline):
    # First call already materialized models; we expose the time it took.
    assert pipeline._cold_start_seconds < 5.0, (
        f"cold-start took {pipeline._cold_start_seconds:.2f}s (>5s)"
    )


def test_health_score_returns_actionable_dict(pipeline):
    h = pipeline.health_score(500001)
    assert isinstance(h, dict)
    assert "health_score" in h
    assert h["action"] in {"Scale", "Continue", "Pivot", "Pause"}
    assert 0.0 <= h["health_score"] <= 100.0


def test_explain_returns_headline_and_counterfactuals(pipeline):
    e = pipeline.explain(500001)
    assert "headline" in e
    assert "counterfactuals" in e
    assert isinstance(e["counterfactuals"], list)


def test_find_similar_returns_k_dicts(pipeline):
    out = pipeline.find_similar(500001, k=5)
    assert isinstance(out, list)
    assert len(out) == 5
    for s in out:
        assert "creative_id" in s
        assert isinstance(s["creative_id"], int)
        # The lookup must not return the query itself
        assert s["creative_id"] != 500001


def test_find_similar_diversify_returns_distinct_ids(pipeline):
    """diversify=True must return k distinct creative_ids that don't repeat
    and don't include the query."""
    out = pipeline.find_similar(500001, k=5, diversify=True)
    assert isinstance(out, list)
    assert len(out) == 5
    ids = [s["creative_id"] for s in out]
    assert len(set(ids)) == len(ids), f"duplicate ids in slate: {ids}"
    assert 500001 not in ids
