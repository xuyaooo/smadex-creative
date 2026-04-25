"""Precompute and cache CLIP embeddings for all creative assets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import yaml

from src.data.loader import DataLoader
from src.embeddings.clip_encoder import CLIPCreativeEncoder, EmbeddingCache


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(config_path).parent
    loader = DataLoader(config_path)
    master_df = loader.load_master_table()

    cache_path = root / cfg["embeddings"]["cache_path"]
    cache = EmbeddingCache(cache_path)

    if cache.exists():
        print(f"Cache already exists at {cache_path}. Delete it to recompute.")
        return

    print(f"Encoding {len(master_df)} creatives...")
    encoder = CLIPCreativeEncoder(
        model_name=cfg["embeddings"]["clip_model"]
    )

    valid_ids, valid_paths = [], []
    for cid in master_df["creative_id"]:
        p = loader.get_asset_path(int(cid))
        if p.exists():
            valid_ids.append(int(cid))
            valid_paths.append(str(p))

    print(f"Found {len(valid_paths)} valid asset files.")
    batch_size = cfg["embeddings"]["batch_size"]
    embeddings = encoder.encode_batch(valid_paths, batch_size=batch_size)

    cache.save(embeddings, valid_ids)
    print(f"Saved embeddings: shape={embeddings.shape} to {cache_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
