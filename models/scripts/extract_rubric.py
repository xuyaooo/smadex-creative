"""
Extract a 15-dim numeric rubric for every creative via OpenRouter (one-time, offline).

The rubric becomes part of the Genome Vector. After this runs, training and
inference no longer call any LLM — they read the cached parquet.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python scripts/extract_rubric.py
  python scripts/extract_rubric.py --dry-run            # 5 creatives, smoke test
  python scripts/extract_rubric.py --model google/gemini-2.0-flash-001
  python scripts/extract_rubric.py --rpm 60             # higher tier

Cost (1,080 creatives):
  google/gemini-2.0-flash-001  ~ $0.10–0.20
  google/gemini-flash-1.5      ~ $0.10
  openai/gpt-4o-mini           ~ $0.30
  anthropic/claude-3-5-haiku   ~ $0.50
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from src.data.loader import DataLoader
from src.training.openrouter_rubric import OpenRouterRubric, RUBRIC_NAMES


JSONL_PATH = "outputs/rubric/rubric_scores.jsonl"
PARQUET_PATH = "outputs/rubric/rubric_scores.parquet"

AVAILABLE_MODELS = [
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-3.5-haiku",
    "qwen/qwen2.5-vl-72b-instruct",
    "amazon/nova-lite-v1",
]


def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    keep = ["creative_id"] + RUBRIC_NAMES
    df = df[[c for c in keep if c in df.columns]].drop_duplicates("creative_id", keep="last")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="google/gemini-2.0-flash-001",
                        choices=AVAILABLE_MODELS)
    parser.add_argument("--rpm", type=int, default=120,
                        help="Per-thread requests/min throttle (only used when --workers=1)")
    parser.add_argument("--workers", type=int, default=64,
                        help="Concurrent API calls (default 64). Set to 1 for serial.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score only 5 creatives to validate setup.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(args.config).parent
    loader = DataLoader(args.config)
    master = loader.load_master_table()
    asset_dir = (root / cfg["data"]["assets_dir"]).resolve()

    cids = master["creative_id"].astype(int).tolist()
    if args.dry_run:
        cids = cids[:5]
        print(f"DRY RUN: 5 creatives")

    print(f"Model:   {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Output:  {root / JSONL_PATH}")

    rubric = OpenRouterRubric(
        model=args.model,
        api_key=args.api_key,
        requests_per_minute=args.rpm,
    )
    rubric.score_all(
        creative_ids=cids,
        asset_dir=asset_dir,
        output_path=root / JSONL_PATH,
        resume=not args.no_resume,
        verbose=True,
        max_workers=args.workers,
    )

    n = jsonl_to_parquet(root / JSONL_PATH, root / PARQUET_PATH)
    print(f"\nWrote {n} rows to {root / PARQUET_PATH}")
    print(f"Rubric dims ({len(RUBRIC_NAMES)}): {RUBRIC_NAMES}")


if __name__ == "__main__":
    main()
