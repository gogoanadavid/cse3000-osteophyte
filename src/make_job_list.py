"""Generate Slurm job-list CSVs from the experiment grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config


def _parse_csv_values(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def ordinal_jobs(
    grid: dict,
    out: str | Path,
    seeds: list[str] | None = None,
    main_curve_only: bool = False,
) -> pd.DataFrame:
    rows = []
    seen = set()
    selected_seeds = [str(seed) for seed in (grid["seeds"] if seeds is None else seeds)]
    main_strategy = grid["main_strategy"]
    for seed_value in selected_seeds:
        seed = int(seed_value) if seed_value.isdigit() else seed_value
        for budget in grid["budgets"]:
            if str(budget) == "0":
                continue
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
        if main_curve_only:
            continue
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


def eval_jobs(ordinal_csv: str | Path, out: str | Path, splits: list[str] | None = None) -> pd.DataFrame:
    jobs = pd.read_csv(ordinal_csv)
    rows = []
    selected_splits = splits or ["val", "test"]
    for _, row in jobs.iterrows():
        for split in selected_splits:
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
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed subset, for example '1,2'. Defaults to all grid seeds.",
    )
    parser.add_argument(
        "--main-curve-only",
        action="store_true",
        help="Write only mixed score-stratified main-curve jobs; skip graded-only and sampling ablations.",
    )
    parser.add_argument(
        "--eval-splits",
        default="val,test",
        help="Comma-separated eval splits for --eval-out. Use 'val' while tuning.",
    )
    args = parser.parse_args()
    grid = load_config(args.experiment_grid)
    df = ordinal_jobs(
        grid,
        args.out,
        seeds=_parse_csv_values(args.seeds),
        main_curve_only=args.main_curve_only,
    )
    print(f"Wrote {len(df)} ordinal jobs to {args.out}")
    if args.eval_out:
        edf = eval_jobs(args.out, args.eval_out, splits=_parse_csv_values(args.eval_splits))
        print(f"Wrote {len(edf)} eval jobs to {args.eval_out}")


if __name__ == "__main__":
    main()
