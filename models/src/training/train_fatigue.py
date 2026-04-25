"""Train the two-stage fatigue detection model."""

import yaml

from src.data.loader import DataLoader
from src.models.fatigue_detector import FatigueDetector


def train(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    loader = DataLoader(config_path)
    master_df = loader.load_master_table()
    daily_df = loader.load_daily_stats()

    print(f"Training on {len(master_df)} creatives, {len(daily_df)} daily rows...")
    fat_cfg = cfg["fatigue_model"]
    detector = FatigueDetector(fat_cfg)
    detector.fit(daily_df, master_df)
    detector.save(fat_cfg)

    print("Fatigue detector saved.")
    n_fatigued = (master_df["creative_status"] == "fatigued").sum()
    print(f"Fatigued creatives in dataset: {n_fatigued}/{len(master_df)}")


if __name__ == "__main__":
    train()
