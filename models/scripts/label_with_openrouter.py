"""
Label all 1,080 creatives using OpenRouter vision API (teacher step).

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python scripts/label_with_openrouter.py

  # Or pass key directly:
  python scripts/label_with_openrouter.py --api-key sk-or-...

  # Pick a different model:
  python scripts/label_with_openrouter.py --model google/gemini-2.0-flash-001

  # Dry run (first 5 creatives only):
  python scripts/label_with_openrouter.py --dry-run

Estimated cost:
  gemini-2.0-flash-001:  ~$0.10  for 1,080 creatives
  gpt-4o-mini:           ~$0.30  for 1,080 creatives
  claude-3-5-haiku:      ~$0.50  for 1,080 creatives
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data.loader import DataLoader
from src.training.openrouter_teacher import OpenRouterTeacher


AVAILABLE_MODELS = [
    "google/gemini-2.5-flash",           # recommended (best quality/$)
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen2.5-vl-72b-instruct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Label creatives with OpenRouter teacher")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key")
    parser.add_argument(
        "--model", default="google/gemini-2.0-flash-001",
        choices=AVAILABLE_MODELS,
        help="OpenRouter vision model to use as teacher",
    )
    parser.add_argument("--rpm", type=int, default=120,
                        help="Per-thread requests per minute (only used when --workers=1)")
    parser.add_argument("--workers", type=int, default=64,
                        help="Concurrent API calls (default 64). Set to 1 for serial.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Label only first 5 creatives to test setup")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (overwrite existing labels)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(args.config).parent
    loader = DataLoader(args.config)
    master_df = loader.load_master_table()
    asset_dir = (root / cfg["data"]["assets_dir"]).resolve()
    output_path = root / cfg["vlm"]["pseudo_labels_path"]

    if args.dry_run:
        master_df = master_df.head(5)
        print(f"DRY RUN: labeling {len(master_df)} creatives with {args.model}")
    else:
        print(f"Labeling {len(master_df)} creatives with {args.model}")

    print(f"Output: {output_path}")
    print(f"RPM: {args.rpm} (delay: {60/args.rpm:.1f}s per request)")

    teacher = OpenRouterTeacher(
        model=args.model,
        api_key=args.api_key,
        requests_per_minute=args.rpm,
    )

    results = teacher.label_all(
        master_df=master_df,
        asset_dir=asset_dir,
        output_path=output_path,
        resume=not args.no_resume,
        verbose=True,
        max_workers=args.workers,
    )

    n_valid = len(results)
    print(f"\nDone. {n_valid} labels written to {output_path}")

    if n_valid > 0:
        # Print a sample
        sample = results[0]
        print("\nSample label:")
        print(f"  creative_id: {sample['creative_id']}")
        print(f"  performance_summary: {sample.get('performance_summary', '')[:120]}")
        print(f"  top_recommendation: {sample.get('top_recommendation', '')[:100]}")

    if n_valid >= len(master_df) * 0.9:
        print(f"\n✓ Labeling complete. Run VLM finetuning next:")
        print("  python scripts/run_pipeline.py --skip-embeddings --skip-tabular --skip-fatigue --skip-teacher")


if __name__ == "__main__":
    main()
