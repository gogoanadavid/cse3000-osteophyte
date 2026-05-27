"""Supervision split construction for weak/mixed/ordinal experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Any

from osteophytes.labels import (
    LOCATION_NAMES,
    NUM_GRADES,
    build_sample_id,
    coerce_grade,
    is_complete_graded_annotation,
    make_sample_id_from_row,
)


SUPERVISION_MODES = ("binary", "mixed", "ordinal")
STRONG_SAMPLING_STRATEGIES = (
    "random",
    "max_grade_stratified",
    "severity_aware",
    "per_location_balanced",
    "grade2_targeted",
)


@dataclass(frozen=True)
class SupervisionSplit:
    strong_sample_ids: set[str]
    weak_sample_ids: set[str]
    rows: list[dict[str, Any]]
    effective_strong_fraction: float
    strong_sampling_strategy: str

    @property
    def strong_count(self) -> int:
        return len(self.strong_sample_ids)

    @property
    def weak_count(self) -> int:
        return len(self.weak_sample_ids)


class MixedSupervisionDataset:
    """Add mixed-supervision metadata without mutating the base dataset index."""

    def __init__(
        self,
        base_dataset: Any,
        strong_sample_ids: set[str],
        weak_sample_ids: set[str] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.strong_sample_ids = set(str(sample_id) for sample_id in strong_sample_ids)
        self.weak_sample_ids = None if weak_sample_ids is None else set(str(sid) for sid in weak_sample_ids)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[idx])
        sample_id = str(sample.get("sample_id") or build_sample_id(sample))
        sample["sample_id"] = sample_id
        sample["is_strong"] = sample_id in self.strong_sample_ids
        if self.weak_sample_ids is not None:
            sample["is_weak"] = sample_id in self.weak_sample_ids
        return sample

    def close(self) -> None:
        close = getattr(self.base_dataset, "close", None)
        if close is not None:
            close()


def _iter_rows(index: Any):
    if hasattr(index, "reset_index") and hasattr(index, "iterrows"):
        yield from index.reset_index(drop=True).iterrows()
    elif hasattr(index, "iterrows"):
        yield from index.iterrows()
    else:
        yield from enumerate(index)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _has_column(row: Any, key: str) -> bool:
    try:
        row[key]
    except (KeyError, TypeError):
        return False
    return True


def _truthy_complete(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def row_grades(row: Any, location_names: tuple[str, ...] = LOCATION_NAMES) -> dict[str, int | None]:
    return {location: coerce_grade(_row_get(row, location), location) for location in location_names}


def row_complete_graded(row: Any, location_names: tuple[str, ...] = LOCATION_NAMES) -> bool:
    if _has_column(row, "complete_graded"):
        trusted = _truthy_complete(_row_get(row, "complete_graded"))
        if trusted is not None:
            return trusted
    return is_complete_graded_annotation({location: _row_get(row, location) for location in location_names})


def max_grade(row: Any, location_names: tuple[str, ...] = LOCATION_NAMES) -> int:
    grades = [grade for grade in row_grades(row, location_names).values() if grade is not None]
    return max(grades) if grades else -1


def positive_location_count(row: Any, location_names: tuple[str, ...] = LOCATION_NAMES) -> int:
    grades = [grade for grade in row_grades(row, location_names).values() if grade is not None]
    return sum(grade > 0 for grade in grades)


def _sample_records(index: Any, location_names: tuple[str, ...] = LOCATION_NAMES) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in _iter_rows(index):
        split = _row_get(row, "split", "train")
        if split is not None and str(split) != "train":
            continue
        sample_id = make_sample_id_from_row(row)
        grades = row_grades(row, location_names)
        complete = row_complete_graded(row, location_names)
        record = {
            "sample_id": sample_id,
            "complete_graded": complete,
            "stratum_max_grade": max(grade for grade in grades.values() if grade is not None) if complete else -1,
            "positive_location_count": sum((grade or 0) > 0 for grade in grades.values()),
            "grades": grades,
        }
        record["stratum"] = (
            f"max_{record['stratum_max_grade']}_pos_{record['positive_location_count']}"
            if complete
            else "incomplete"
        )
        records.append(record)
    return records


def _target_count(num_candidates: int, fraction: float) -> int:
    if num_candidates <= 0 or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return num_candidates
    return min(num_candidates, max(1, int(math.floor(num_candidates * fraction + 0.5))))


def _sample_uniform(records: list[dict[str, Any]], target_count: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[:target_count]


def _sample_max_grade_stratified(
    records: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["stratum_max_grade"]), []).append(record)
    for group in grouped.values():
        rng.shuffle(group)

    allocations: dict[int, int] = {}
    remainders: dict[int, float] = {}
    fraction = target_count / len(records)
    for grade, group in grouped.items():
        raw = len(group) * fraction
        allocations[grade] = int(math.floor(raw))
        remainders[grade] = raw - allocations[grade]
    remaining = target_count - sum(allocations.values())
    for grade in sorted(grouped, key=lambda key: (-remainders[key], key)):
        if remaining <= 0:
            break
        if allocations[grade] < len(grouped[grade]):
            allocations[grade] += 1
            remaining -= 1

    selected: list[dict[str, Any]] = []
    for grade, group in grouped.items():
        selected.extend(group[: allocations[grade]])
    return selected[:target_count]


def _sample_severity_aware(
    records: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    high = [record for record in records if any((grade or 0) >= 2 for grade in record["grades"].values())]
    low = [record for record in records if record not in high]
    rng.shuffle(high)
    rng.shuffle(low)
    return (high + low)[:target_count]


def _coverage_score(record: dict[str, Any], counts: dict[tuple[str, int], int]) -> float:
    score = 0.0
    for location, grade in record["grades"].items():
        if grade is None:
            continue
        grade_weight = {0: 0.25, 1: 1.0, 2: 3.0, 3: 4.0}[int(grade)]
        score += grade_weight / (1.0 + counts[(location, int(grade))])
    return score


def _sample_per_location_balanced(
    records: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    remaining = list(records)
    rng.shuffle(remaining)
    selected: list[dict[str, Any]] = []
    counts = {(location, grade): 0 for location in LOCATION_NAMES for grade in range(NUM_GRADES)}
    while remaining and len(selected) < target_count:
        best_index = max(
            range(len(remaining)),
            key=lambda idx: (_coverage_score(remaining[idx], counts), -idx),
        )
        record = remaining.pop(best_index)
        selected.append(record)
        for location, grade in record["grades"].items():
            if grade is not None:
                counts[(location, int(grade))] += 1
    return selected


def _sample_grade2_targeted(
    records: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Guarantee per-location coverage of grade-2 and grade-3 samples."""
    from osteophytes.labels import LOCATION_NAMES

    selected_ids: set[str] = set()
    priority: list[dict[str, Any]] = []

    for location in LOCATION_NAMES:
        grade3_here = [
            record
            for record in records
            if (record["grades"].get(location) or 0) == 3
            and record["sample_id"] not in selected_ids
        ]
        rng.shuffle(grade3_here)
        for record in grade3_here:
            selected_ids.add(record["sample_id"])
            priority.append(record)

        grade2_here = [
            record
            for record in records
            if (record["grades"].get(location) or 0) == 2
            and record["sample_id"] not in selected_ids
        ]
        rng.shuffle(grade2_here)
        for record in grade2_here:
            selected_ids.add(record["sample_id"])
            priority.append(record)

    if len(priority) >= target_count:
        return priority[:target_count]

    lower = [record for record in records if record["sample_id"] not in selected_ids]
    rng.shuffle(lower)

    remaining_budget = target_count - len(priority)
    return priority + lower[:remaining_budget]


def _select_strong_records(
    candidates: list[dict[str, Any]],
    fraction: float,
    seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    target_count = _target_count(len(candidates), fraction)
    rng = random.Random(seed)
    if target_count == 0:
        return []
    if strategy == "random":
        return _sample_uniform(candidates, target_count, rng)
    if strategy == "max_grade_stratified":
        return _sample_max_grade_stratified(candidates, target_count, rng)
    if strategy == "severity_aware":
        return _sample_severity_aware(candidates, target_count, rng)
    if strategy == "per_location_balanced":
        return _sample_per_location_balanced(candidates, target_count, rng)
    if strategy == "grade2_targeted":
        return _sample_grade2_targeted(candidates, target_count, rng)
    raise ValueError(f"Unsupported strong sampling strategy: {strategy}")


def build_supervision_split(
    index: Any,
    supervision_mode: str,
    strong_fraction: float,
    seed: int,
    strategy: str = "random",
    ordinal_include_weak: bool = False,
) -> SupervisionSplit:
    """Build a deterministic supervision split from train rows only."""
    if supervision_mode not in SUPERVISION_MODES:
        raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
    if strategy not in STRONG_SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported strong sampling strategy: {strategy}")
    if not 0.0 <= strong_fraction <= 1.0:
        raise ValueError("strong_fraction must be in [0, 1]")

    records = _sample_records(index)
    complete_records = [record for record in records if record["complete_graded"]]
    if supervision_mode == "binary":
        effective_fraction = 0.0
        strong_records: list[dict[str, Any]] = []
    elif supervision_mode == "ordinal":
        effective_fraction = 1.0
        strong_records = list(complete_records)
    else:
        effective_fraction = strong_fraction
        strong_records = _select_strong_records(complete_records, strong_fraction, seed, strategy)

    strong_ids = {str(record["sample_id"]) for record in strong_records}
    weak_ids: set[str] = set()
    split_rows: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        is_strong = sample_id in strong_ids
        if supervision_mode == "ordinal" and not ordinal_include_weak and not is_strong:
            supervision = "ignored"
        else:
            supervision = "strong" if is_strong else "weak"
        if supervision == "weak":
            weak_ids.add(sample_id)
        split_rows.append(
            {
                "sample_id": sample_id,
                "complete_graded": bool(record["complete_graded"]),
                "stratum": record["stratum"],
                "stratum_max_grade": record["stratum_max_grade"],
                "positive_location_count": record["positive_location_count"],
                "is_strong": is_strong,
                "is_weak": supervision == "weak",
                "supervision": supervision,
            }
        )
    return SupervisionSplit(
        strong_sample_ids=strong_ids,
        weak_sample_ids=weak_ids,
        rows=split_rows,
        effective_strong_fraction=effective_fraction,
        strong_sampling_strategy=strategy,
    )


def first_sample_ids_from_split(split_rows: list[dict[str, Any]], count: int = 3) -> list[str]:
    return [str(row["sample_id"]) for row in split_rows[:count]]


def grade_distribution_rows(
    index: Any,
    strong_sample_ids: set[str],
    strong_fraction: float,
    seed: int,
    strategy: str,
    location_names: tuple[str, ...] = LOCATION_NAMES,
) -> list[dict[str, Any]]:
    counts = {(location, grade): 0 for location in location_names for grade in range(NUM_GRADES)}
    totals = {location: 0 for location in location_names}
    for _, row in _iter_rows(index):
        if str(_row_get(row, "split", "train")) != "train":
            continue
        sample_id = make_sample_id_from_row(row)
        if sample_id not in strong_sample_ids:
            continue
        for location in location_names:
            grade = coerce_grade(_row_get(row, location), location)
            if grade is None:
                continue
            counts[(location, grade)] += 1
            totals[location] += 1

    rows: list[dict[str, Any]] = []
    for location in location_names:
        total = totals[location]
        for grade in range(NUM_GRADES):
            count = counts[(location, grade)]
            rows.append(
                {
                    "location": location,
                    "grade": grade,
                    "count": count,
                    "fraction": count / total if total > 0 else 0.0,
                    "strong_fraction": strong_fraction,
                    "seed": seed,
                    "strong_sampling_strategy": strategy,
                }
            )
    return rows


def high_grade_coverage_rows(
    index: Any,
    strong_sample_ids: set[str],
    seed: int,
    strategy: str,
    location_names: tuple[str, ...] = LOCATION_NAMES,
) -> list[dict[str, Any]]:
    selected_rows = []
    for _, row in _iter_rows(index):
        if str(_row_get(row, "split", "train")) != "train":
            continue
        sample_id = make_sample_id_from_row(row)
        if sample_id in strong_sample_ids:
            selected_rows.append(row)

    rows: list[dict[str, Any]] = []
    for location in location_names:
        grades = [coerce_grade(_row_get(row, location), location) for row in selected_rows]
        grades = [grade for grade in grades if grade is not None]
        total = len(grades)
        grade2_count = sum(grade == 2 for grade in grades)
        grade3_count = sum(grade == 3 for grade in grades)
        rows.append(
            {
                "location": location,
                "selected_count": total,
                "grade2_count": grade2_count,
                "grade3_count": grade3_count,
                "grade2_or_3_count": grade2_count + grade3_count,
                "grade3_fraction": grade3_count / total if total else 0.0,
                "grade2_or_3_fraction": (grade2_count + grade3_count) / total if total else 0.0,
                "seed": seed,
                "strong_sampling_strategy": strategy,
            }
        )
    any_high = 0
    for row in selected_rows:
        grades = row_grades(row, location_names)
        if any((grade or 0) >= 2 for grade in grades.values()):
            any_high += 1
    rows.append(
        {
            "location": "any_location",
            "selected_count": len(selected_rows),
            "grade2_count": math.nan,
            "grade3_count": math.nan,
            "grade2_or_3_count": any_high,
            "grade3_fraction": math.nan,
            "grade2_or_3_fraction": any_high / len(selected_rows) if selected_rows else 0.0,
            "seed": seed,
            "strong_sampling_strategy": strategy,
        }
    )
    return rows


def save_supervision_artifacts(
    output_dir: str | Path,
    split: SupervisionSplit,
    train_index: Any,
    seed: int,
) -> None:
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": sorted(split.strong_sample_ids)}).to_csv(
        output_path / "strong_sample_ids.csv",
        index=False,
    )
    pd.DataFrame(split.rows).to_csv(output_path / "supervision_split.csv", index=False)
    pd.DataFrame(
        grade_distribution_rows(
            train_index,
            split.strong_sample_ids,
            split.effective_strong_fraction,
            seed,
            split.strong_sampling_strategy,
        )
    ).to_csv(output_path / "strong_grade_distribution_by_location.csv", index=False)
    pd.DataFrame(
        high_grade_coverage_rows(
            train_index,
            split.strong_sample_ids,
            seed,
            split.strong_sampling_strategy,
        )
    ).to_csv(output_path / "strong_high_grade_coverage_summary.csv", index=False)


def assert_startup_sanity(
    split: SupervisionSplit,
    train_dataset: Any,
    wrapped_train_dataset: Any,
    strong_fraction: float,
) -> dict[str, Any]:
    """Return and print basic sanity information for reproducible training starts."""
    first_split_ids = first_sample_ids_from_split(split.rows)
    if hasattr(train_dataset, "index"):
        first_wrapped_ids = [
            make_sample_id_from_row(row)
            for _, row in list(_iter_rows(train_dataset.index))[: min(3, len(wrapped_train_dataset))]
        ]
    else:
        first_wrapped_ids = [
            str(wrapped_train_dataset[i]["sample_id"]) for i in range(min(3, len(wrapped_train_dataset)))
        ]
    if strong_fraction > 0 and split.strong_count > 0:
        found_strong = any(bool(row["is_strong"]) for row in split.rows)
        if not found_strong:
            raise RuntimeError("No wrapped training sample is marked strong despite a nonzero strong split.")
    summary = {
        "first_split_sample_ids": first_split_ids,
        "first_wrapped_sample_ids": first_wrapped_ids,
        "strong_count": split.strong_count,
        "weak_count": split.weak_count,
        "train_count": len(train_dataset),
    }
    print(f"First 3 split sample IDs: {first_split_ids}")
    print(f"First 3 wrapped sample IDs: {first_wrapped_ids}")
    print(f"Strong train samples: {split.strong_count}")
    print(f"Weak train samples: {split.weak_count}")
    return summary
