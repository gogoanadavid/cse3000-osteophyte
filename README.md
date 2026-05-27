# Osteophyte Severity Estimation

Minimal project skeleton for a TU Delft CSE3000 research project on osteophyte
severity estimation in hip X-rays.

This repository currently contains data discovery, label utilities, and audit
scaffolding only. It does not contain training code, model code, or copied data.

## DelftBlue Data Policy

The project is intended to run on DelftBlue. The image data and annotations are
confidential and must not be copied into this local repository. Scripts should
read from configured DelftBlue paths and write derived audit outputs to scratch.

Default paths are stored in `configs/delftblue.yaml`:

```yaml
project_root: /tudelft.net/staff-umbrella/osteoarthritis2024/project/bsc2026/
source_data_root: /tudelft.net/staff-umbrella/osteoarthritis2024/shared/data/
scratch_root: /scratch/dgogoana/osteophytes_project/
```

## Structure

```text
configs/
  delftblue.yaml
scripts/
  00_discover_data.py
  01_audit_labels_and_splits.py
  02_build_dataset_index.py
  03_make_visual_audit_sheet.py
  04_smoke_dataloader.py
  05_train_binary_baseline.py
src/
  osteophytes/
    config.py
    data_index.py
    dataset.py
    labels.py
    metrics.py
    models.py
experiments/
  01_binary_baseline/
tests/
  test_labels.py
```

## Usage

Discover candidate files on DelftBlue:

```bash
python scripts/00_discover_data.py
```

Audit a label CSV:

```bash
python scripts/01_audit_labels_and_splits.py --csv-path /path/to/labels.csv
```

Optionally provide a split/list file path. Subject ID and split validation are
left as TODOs until the exact CSV and split schemas are confirmed.

Build the usable H5-backed dataset index on DelftBlue:

```bash
python scripts/02_build_dataset_index.py
```

By default this writes `dataset_index.csv` and `dataset_index_summary.json` to
`/scratch/dgogoana/osteophytes_project/audits/`.

Create visual audit sheets from selected H5-backed examples:

```bash
python scripts/03_make_visual_audit_sheet.py
```

By default this reads `/scratch/dgogoana/osteophytes_project/audits/dataset_index.csv`
and writes one PNG grid per osteophyte location under
`/scratch/dgogoana/osteophytes_project/audits/visual_audit/`.

Smoke-test a single PyTorch DataLoader batch:

```bash
python scripts/04_smoke_dataloader.py
```

Run the first binary-supervised baseline:

```bash
python scripts/05_train_binary_baseline.py --smoke
```

## Tests

```bash
pytest
```
