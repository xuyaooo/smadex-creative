from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml


class DataLoader:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["data"]
        self.root = Path(config_path).parent

    def _path(self, key: str) -> Path:
        return (self.root / self.cfg[key]).resolve()

    def load_master_table(self) -> pd.DataFrame:
        summary = pd.read_csv(self._path("creative_summary"))
        campaigns = pd.read_csv(self._path("campaigns"))
        advertisers = pd.read_csv(self._path("advertisers"))

        df = summary.merge(
            campaigns[["campaign_id", "advertiser_id", "objective", "primary_theme",
                        "target_age_segment", "target_os", "kpi_goal", "daily_budget_usd",
                        "start_date", "end_date"]],
            on="campaign_id", how="left"
        ).merge(
            advertisers[["advertiser_id", "hq_region"]],
            on="advertiser_id", how="left"
        )

        df["campaign_duration"] = (
            pd.to_datetime(df["end_date"]) - pd.to_datetime(df["start_date"])
        ).dt.days

        return df

    def load_daily_stats(self) -> pd.DataFrame:
        df = pd.read_csv(self._path("daily_stats"), parse_dates=["date"])
        return df

    def get_creative_timeseries(self, creative_id: int, daily_df: pd.DataFrame) -> pd.DataFrame:
        return daily_df[daily_df["creative_id"] == creative_id].sort_values("date").reset_index(drop=True)

    def get_asset_path(self, creative_id: int) -> Path:
        return (self.root / self.cfg["assets_dir"] / f"creative_{creative_id}.png").resolve()

    def split_train_val_test(
        self, df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.10, seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        campaign_ids = df["campaign_id"].unique()
        rng.shuffle(campaign_ids)

        n = len(campaign_ids)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        test_ids = set(campaign_ids[:n_test])
        val_ids = set(campaign_ids[n_test: n_test + n_val])

        test_df = df[df["campaign_id"].isin(test_ids)].reset_index(drop=True)
        val_df = df[df["campaign_id"].isin(val_ids)].reset_index(drop=True)
        train_df = df[~df["campaign_id"].isin(test_ids | val_ids)].reset_index(drop=True)

        return train_df, val_df, test_df
