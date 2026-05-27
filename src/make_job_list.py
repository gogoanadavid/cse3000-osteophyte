"""Generate Slurm job-list CSVs from the experiment grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config


def ordinal_jobs(grid: dict, out: str | Path) -> pd.DataFrame:
    rows = []
    seen = set()
    seeds = grid["seeds"]
    main_strategy = grid["main_strategy"]
    for seed in seeds:
        for budget in grid["budgets"]:
            key = (seed, main_strategy, budget, "mixed")
            if key not in seen:
                seen.add(key)
                rows.append(
                    {
                        "seed": seed,
                        "strategy": main_strategy,
                        "budget_name": budget,
                        "mode": "mixed",
                        "budget_file": f"budgets/{main_strategy}_seed{seed}/budget_{budget}.csv",
                        "binary_checkpoint": f"outputs/checkpoints/binary_seed{seed}/best.pt",
                        "out_dir": f"outputs/checkpoints/ordinal/{main_strategy}_seed{seed}_budget{budget}",
                    }
                )
        for budget in grid["graded_only_budgets"]:
            key = (seed, main_strategy, budget, "graded_only")
            if key not in seen:
                seen.add(key)
                rows.append(
                    {
                        "seed": seed,
                        "strategy": main_strategy,
                        "budget_name": budget,
                        "mode": "graded_only",
                        "budget_file": f"budgets/{main_strategy}_seed{seed}/budget_{budget}.csv",
                        "binary_checkpoint": f"outputs/checkpoints/binary_seed{seed}/best.pt",
                        "out_dir": f"outputs/checkpoints/ordinal/graded_only_{main_strategy}_seed{seed}_budget{budget}",
                    }
                )
        for strategy in grid["strategies"]:
            for budget in grid["sampling_comparison_budgets"]:
                key = (seed, strategy, budget, "mixed")
                if key not in seen:
                    seen.add(key)
                    rows.append(
                        {
                            "seed": seed,
                            "strategy": strategy,
                            "budget_name": budget,
                            "mode": "mixed",
                            "budget_file": f"budgets/{strategy}_seed{seed}/budget_{budget}.csv",
                            "binary_checkpoint": f"outputs/checkpoints/binary_seed{seed}/best.pt",
                            "out_dir": f"outputs/checkpoints/ordinal/{strategy}_seed{seed}_budget{budget}",
                        }
                    )
    df = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def eval_jobs(ordinal_csv: str | Path, out: str | Path) -> pd.DataFrame:
    jobs = pd.read_csv(ordinal_csv)
    rows = []
    for _, row in jobs.iterrows():
        for split in ["val", "test"]:
            rows.append(
                {
                    "checkpoint": f"{row['out_dir']}/best.pt",
                    "split": split,
                    "out_dir": f"outputs/metrics/{Path(str(row['out_dir'])).name}_{split}",
                }
            )
    df = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-grid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-out", default=None)
    args = parser.parse_args()
    grid = load_config(args.experiment_grid)
    df = ordinal_jobs(grid, args.out)
    print(f"Wrote {len(df)} ordinal jobs to {args.out}")
    if args.eval_out:
        edf = eval_jobs(args.out, args.eval_out)
        print(f"Wrote {len(edf)} eval jobs to {args.eval_out}")


if __name__ == "__main__":
    main()
