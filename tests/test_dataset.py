from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_h5_dataset_returns_expected_sample(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    h5py = pytest.importorskip("h5py")
    torch = pytest.importorskip("torch")

    from osteophytes.dataset import HipOsteophyteDataset

    h5_path = tmp_path / "images.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("scans/s1/00/L/image", data=np.arange(224 * 224, dtype=np.float32).reshape(224, 224))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "cohort": "fake",
                "subject": "s1",
                "visit": "00",
                "side": "L",
                "split": "train",
                "h5_internal_path": "scans/s1/00/L/image",
                "osteo_acet_inf": 0,
                "osteo_acet_sup": 1,
                "osteo_fem_inf": 2,
                "osteo_fem_sup": 3,
                "osteo_acet_inf_binary": 0,
                "osteo_acet_sup_binary": 1,
                "osteo_fem_inf_binary": 1,
                "osteo_fem_sup_binary": 1,
                "complete_graded": True,
            }
        ]
    ).to_csv(index_path, index=False)

    dataset = HipOsteophyteDataset(index_path, h5_path, split="train")
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (1, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert float(sample["image"].min()) >= 0.0
    assert float(sample["image"].max()) <= 1.0
    assert tuple(sample["graded_labels"].shape) == (4,)
    assert tuple(sample["binary_labels"].shape) == (4,)
    assert sample["sample_id"] == "s1|00|L"
    assert "::" not in sample["sample_id"]
    dataset.close()
