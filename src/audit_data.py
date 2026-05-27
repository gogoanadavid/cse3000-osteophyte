"""Audit index CSV label distributions and split leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _locations(df: pd.DataFrame) -> list[str]:
    return [col.removeprefix("binary_") for col in df.columns if col.startswith("binary_")]


def audit_index(index_csv: str | Path, out_dir: str | Path) -> dict[str, pd.DataFrame | str]:
    df = pd.read_csv(index_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    locations = _locations(df)

    split_counts = df.groupby("split").size().rename("n").reset_index()
    split_counts.to_csv(out_dir / "split_counts.csv", index=False)

    complete_counts = (
        df.groupby("split")["has_complete_grades"]
        .agg(total="count", complete="sum")
        .reset_index()
    )
    complete_counts["binary_only"] = complete_counts["total"] - complete_counts["complete"]
    complete_counts.to_csv(out_dir / "grade_availability.csv", index=False)

    grade_rows = []
    binary_rows = []
    grade3_rows = []
    for split, sdf in df.groupby("split"):
        for loc in locations:
            grade_counts = sdf[f"grade_{loc}"].value_counts(dropna=False).to_dict()
            binary_counts = sdf[f"binary_{loc}"].value_counts(dropna=False).to_dict()
            for label, count in sorted(grade_counts.items()):
                grade_rows.append({"split": split, "location": loc, "grade": int(label), "count": int(count)})
            for label, count in sorted(binary_counts.items()):
                binary_rows.append({"split": split, "location": loc, "binary": int(label), "count": int(count)})
            grade3_rows.append(
                {
                    "split": split,
                    "location": loc,
                    "grade3_count": int((sdf[f"grade_{loc}"] == 3).sum()),
                }
            )
    grade_dist = pd.DataFrame(grade_rows)
    binary_dist = pd.DataFrame(binary_rows)
    grade3 = pd.DataFrame(grade3_rows)
    grade_dist.to_csv(out_dir / "grade_distribution.csv", index=False)
    binary_dist.to_csv(out_dir / "binary_distribution.csv", index=False)
    grade3.to_csv(out_dir / "grade3_counts.csv", index=False)

    leakage_lines: list[str] = []
    split_subjects = {
        split: set(sdf["subject_id"].astype(str).tolist()) for split, sdf in df.groupby("split")
    }
    splits = sorted(split_subjects)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = sorted(split_subjects[a] & split_subjects[b])
            if overlap:
                leakage_lines.append(
                    f"VERY OBVIOUS WARNING: subject_id leakage between {a} and {b}: "
                    f"{len(overlap)} subjects. Examples: {overlap[:10]}"
                )
    if not leakage_lines:
        leakage_lines.append("No subject_id overlap detected between splits.")

    warning_file = Path("outputs/logs/build_index_warnings.txt")
    warning_summary = warning_file.read_text(encoding="utf-8") if warning_file.exists() else ""
    report = []
    report.append("Split counts\n============")
    report.append(split_counts.to_string(index=False))
    report.append("\nGrade availability\n==================")
    report.append(complete_counts.to_string(index=False))
    report.append("\nSubject leakage\n===============")
    report.extend(leakage_lines)
    report.append("\nGrade-3 counts\n==============")
    report.append(grade3.to_string(index=False))
    report.append("\nConsistency warning summary\n===========================")
    report.append(warning_summary if warning_summary.strip() else "No build-index consistency warnings recorded.")
    text = "\n".join(report) + "\n"
    (out_dir / "audit_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return {
        "split_counts": split_counts,
        "grade_availability": complete_counts,
        "grade_distribution": grade_dist,
        "binary_distribution": binary_dist,
        "report": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    audit_index(args.index, args.out_dir)


if __name__ == "__main__":
    main()
