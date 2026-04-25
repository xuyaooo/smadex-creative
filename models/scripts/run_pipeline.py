"""
End-to-end training pipeline.

Steps:
  1. Precompute CLIP embeddings
  2. Train tabular model (XGBoost + CLIP)
  3. Train fatigue detector (LightGBM)
  4. Generate teacher pseudo-labels (SDFT step 1)
  5. Finetune student VLM with LoRA (SDFT step 2)
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def run_step(name: str, script: str, config: str, skip: bool = False) -> None:
    if skip:
        print(f"[SKIP] {name}")
        return
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script, "--config", config],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {name} failed (exit code {result.returncode})")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Creative Intelligence pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-fatigue", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher pseudo-labeling (requires GPU with ~8GB VRAM)")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM finetuning (requires GPU with ~6GB VRAM)")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    src_dir = scripts_dir.parent / "src" / "training"

    run_step(
        "1. Precompute CLIP embeddings",
        str(scripts_dir / "precompute_embeddings.py"),
        args.config,
        skip=args.skip_embeddings,
    )
    run_step(
        "2. Train tabular model (XGBoost + CLIP)",
        str(src_dir / "train_tabular.py"),
        args.config,
        skip=args.skip_tabular,
    )
    run_step(
        "3. Train fatigue detector (LightGBM)",
        str(src_dir / "train_fatigue.py"),
        args.config,
        skip=args.skip_fatigue,
    )
    run_step(
        "4. Generate teacher pseudo-labels (SDFT step 1)",
        str(scripts_dir / "generate_pseudo_labels.py"),
        args.config,
        skip=args.skip_teacher,
    )

    if not args.skip_vlm:
        from src.data.loader import DataLoader
        from src.training.train_vlm import VLMFinetuner, VLMCreativeDataset
        import yaml as _yaml
        from pathlib import Path as _Path

        with open(args.config) as f:
            cfg = _yaml.safe_load(f)

        root = _Path(args.config).parent
        loader = DataLoader(args.config)
        master_df = loader.load_master_table()

        labels_path = root / cfg["vlm"]["pseudo_labels_path"]
        if not labels_path.exists():
            print("No pseudo-labels found — skipping VLM finetuning.")
        else:
            finetuner = VLMFinetuner(cfg["vlm"])
            labels = finetuner.load_labels(str(labels_path))

            from src.data.feature_engineering import TabularFeatureEngineer
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(cfg["vlm"]["student_model"])
            asset_dir = (root / cfg["data"]["assets_dir"]).resolve()

            train_df, val_df, _ = loader.split_train_val_test(master_df)
            train_dataset = VLMCreativeDataset(train_df, labels, asset_dir, processor,
                                               max_length=cfg["vlm"]["max_seq_length"])
            val_dataset = VLMCreativeDataset(val_df, labels, asset_dir, processor,
                                             max_length=cfg["vlm"]["max_seq_length"])

            output_dir = str(root / cfg["vlm"]["student_checkpoint"])
            print(f"\nFinetuning student VLM: {cfg['vlm']['student_model']}")
            print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
            finetuner.train(train_dataset, val_dataset, output_dir)
            print(f"Student VLM saved to {output_dir}")

    print("\nPipeline complete. Run: python demo/app.py")


if __name__ == "__main__":
    main()
