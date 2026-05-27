from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.supervision import MixedSupervisionDataset, build_supervision_split


class FakeIndex:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def reset_index(self, drop: bool = False) -> "FakeIndex":
        return self

    def iterrows(self):
        yield from enumerate(self.rows)


class FakeBaseDataset:
    def __init__(self) -> None:
        self.samples = [
            {"subject": "s1", "visit": "v0", "side": "L", "value": 1},
            {"subject": "s2", "visit": "v1", "side": "R", "value": 2},
        ]
        self.closed = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.samples[idx]

    def close(self) -> None:
        self.closed = True


def make_row(row_index: int, grades: tuple[int | None, int | None, int | None, int | None]) -> dict[str, object]:
    row: dict[str, object] = {
        "subject": f"subject_{row_index:03d}",
        "visit": "00",
        "side": "L" if row_index % 2 == 0 else "R",
        "split": "train",
        "osteo_acet_inf": grades[0],
        "osteo_acet_sup": grades[1],
        "osteo_fem_inf": grades[2],
        "osteo_fem_sup": grades[3],
    }
    row["complete_graded"] = all(grade is not None for grade in grades)
    return row


def make_index(num_rows: int, incomplete_every: int | None = None) -> FakeIndex:
    rows = []
    for row_index in range(num_rows):
        grade = row_index % 4
        grades: tuple[int | None, int | None, int | None, int | None] = (
            grade,
            (grade + 1) % 4,
            (grade + 2) % 4,
            (grade + 3) % 4,
        )
        if incomplete_every is not None and row_index % incomplete_every == 0:
            grades = (grades[0], grades[1], grades[2], None)
        rows.append(make_row(row_index, grades))
    return FakeIndex(rows)


def count_high(split, index: FakeIndex) -> int:
    rows_by_id = {f"{row['subject']}|{row['visit']}|{row['side']}": row for row in index.rows}
    count = 0
    for sample_id in split.strong_sample_ids:
        row = rows_by_id[sample_id]
        if any(int(row[column]) >= 2 for column in ("osteo_acet_inf", "osteo_acet_sup", "osteo_fem_inf", "osteo_fem_sup")):
            count += 1
    return count


def high_bin_coverage(split, index: FakeIndex) -> int:
    rows_by_id = {f"{row['subject']}|{row['visit']}|{row['side']}": row for row in index.rows}
    bins = set()
    for sample_id in split.strong_sample_ids:
        row = rows_by_id[sample_id]
        for location in ("osteo_acet_inf", "osteo_acet_sup", "osteo_fem_inf", "osteo_fem_sup"):
            grade = int(row[location])
            if grade >= 2:
                bins.add((location, grade))
    return len(bins)


def test_random_split_is_deterministic_and_uses_pipe_ids() -> None:
    index = make_index(40)

    split_a = build_supervision_split(index, "mixed", 0.25, 123, strategy="random")
    split_b = build_supervision_split(index, "mixed", 0.25, 123, strategy="random")

    assert split_a.strong_sample_ids == split_b.strong_sample_ids
    assert split_a.rows == split_b.rows
    assert len(split_a.strong_sample_ids) == 10
    assert all("|" in sample_id for sample_id in split_a.strong_sample_ids)
    assert all("::" not in sample_id for sample_id in split_a.strong_sample_ids)


def test_mixed_supervision_dataset_adds_flags_without_mutating_base() -> None:
    base_dataset = FakeBaseDataset()
    wrapped = MixedSupervisionDataset(base_dataset, {"s1|v0|L"}, weak_sample_ids={"s2|v1|R"})

    strong_sample = wrapped[0]
    weak_sample = wrapped[1]

    assert strong_sample["sample_id"] == "s1|v0|L"
    assert strong_sample["is_strong"] is True
    assert strong_sample["is_weak"] is False
    assert weak_sample["sample_id"] == "s2|v1|R"
    assert weak_sample["is_strong"] is False
    assert weak_sample["is_weak"] is True
    assert "sample_id" not in base_dataset.samples[0]
    wrapped.close()
    assert base_dataset.closed


def test_supervision_modes_and_complete_graded_constraint() -> None:
    index = make_index(12, incomplete_every=3)

    binary = build_supervision_split(index, "binary", 0.50, 123)
    mixed = build_supervision_split(index, "mixed", 1.0, 123)
    ordinal = build_supervision_split(index, "ordinal", 0.05, 123)
    incomplete_ids = {row["sample_id"] for row in mixed.rows if not row["complete_graded"]}

    assert binary.effective_strong_fraction == 0.0
    assert len(binary.strong_sample_ids) == 0
    assert mixed.effective_strong_fraction == 1.0
    assert ordinal.effective_strong_fraction == 1.0
    assert mixed.strong_sample_ids.isdisjoint(incomplete_ids)
    assert ordinal.strong_sample_ids.isdisjoint(incomplete_ids)


def test_severity_aware_includes_more_high_grade_samples_than_random() -> None:
    rows = [make_row(i, (0, 0, 0, 0)) for i in range(20)]
    rows.extend(make_row(20 + i, (2 + (i % 2), 0, 0, 0)) for i in range(8))
    index = FakeIndex(rows)

    random_split = build_supervision_split(index, "mixed", 0.25, 7, strategy="random")
    severity_split = build_supervision_split(index, "mixed", 0.25, 7, strategy="severity_aware")

    assert count_high(severity_split, index) >= count_high(random_split, index)
    assert count_high(severity_split, index) == len(severity_split.strong_sample_ids)


def test_per_location_balanced_improves_high_grade_bin_coverage() -> None:
    rows = []
    high_patterns = [
        (3, 0, 0, 0),
        (0, 3, 0, 0),
        (0, 0, 3, 0),
        (0, 0, 0, 3),
        (2, 0, 0, 0),
        (0, 2, 0, 0),
        (0, 0, 2, 0),
        (0, 0, 0, 2),
    ]
    rows.extend(make_row(i, pattern) for i, pattern in enumerate(high_patterns))
    rows.extend(make_row(100 + i, (0, 0, 0, 0)) for i in range(24))
    index = FakeIndex(rows)

    random_split = build_supervision_split(index, "mixed", 0.25, 11, strategy="random")
    balanced_split = build_supervision_split(index, "mixed", 0.25, 11, strategy="per_location_balanced")

    assert high_bin_coverage(balanced_split, index) >= high_bin_coverage(random_split, index)
