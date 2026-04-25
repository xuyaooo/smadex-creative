"""Force-recompute the visual-encoder embedding cache.

Used after swapping the visual encoder (e.g. CLIP ViT-B/32 -> SigLIP 2 base).
Deletes the existing cache and regenerates it with the model configured in
config.yaml:embeddings.clip_model. Uses GPU if available.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data.loader import DataLoader
from src.embeddings.clip_encoder import CLIPCreativeEncoder, EmbeddingCache


def main(config_path: str = "config.yaml") -> None:
    cfg_path = Path(config_path).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    root = cfg_path.parent
    loader = DataLoader(str(cfg_path))
    master_df = loader.load_master_table()

    cache_path = root / cfg["embeddings"]["cache_path"]
    cache = EmbeddingCache(cache_path)

    if cache.exists():
        print(f"Removing existing cache at {cache_path}")
        cache_path.unlink()

    model_name = cfg["embeddings"]["clip_model"]
    print(f"Loading visual encoder: {model_name}")
    encoder = CLIPCreativeEncoder(model_name=model_name, device="auto")
    print(f"Using device: {encoder.device}")

    valid_ids, valid_paths = [], []
    for cid in master_df["creative_id"]:
        p = loader.get_asset_path(int(cid))
        if p.exists():
            valid_ids.append(int(cid))
            valid_paths.append(str(p))

    print(f"Encoding {len(valid_paths)} creatives at batch_size={cfg['embeddings']['batch_size']}")
    t0 = time.time()
    embeddings = encoder.encode_batch(valid_paths, batch_size=cfg["embeddings"]["batch_size"])
    elapsed = time.time() - t0
    print(f"Encoded in {elapsed:.1f}s -> shape={embeddings.shape}, dtype={embeddings.dtype}")

    cache.save(embeddings, valid_ids)
    print(f"Saved embeddings to {cache_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
