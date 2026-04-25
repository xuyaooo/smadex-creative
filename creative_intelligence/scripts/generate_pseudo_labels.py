"""Generate teacher pseudo-labels for all creatives (SDFT step 1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data.loader import DataLoader
from src.training.teacher_labeling import TeacherLabeler


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(config_path).parent
    vlm_cfg = cfg["vlm"]

    loader = DataLoader(config_path)
    master_df = loader.load_master_table()
    asset_dir = (root / cfg["data"]["assets_dir"]).resolve()
    output_path = root / vlm_cfg["pseudo_labels_path"]

    print(f"Teacher model: {vlm_cfg['teacher_model']}")
    print(f"Labeling {len(master_df)} creatives...")
    print(f"Output: {output_path}")

    labeler = TeacherLabeler(model_name=vlm_cfg["teacher_model"])
    labels = labeler.label_all_creatives(
        master_df=master_df,
        asset_dir=asset_dir,
        output_path=output_path,
        resume=True,
    )
    print(f"Generated {len(labels)} new labels.")

    # Validate
    valid = [l for l in labels if labeler.validate_label(l)]
    print(f"Valid labels: {len(valid)}/{len(labels)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
