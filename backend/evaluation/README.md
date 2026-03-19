# Evaluation Folder Organization

This folder is organized by purpose:

- `queries/`
  - Query sets used for evaluation runs.

- `data/labeling_csv/`
  - Raw chunk export CSVs used for manual relevance labeling.

- `data/labeled_xlsx/`
  - Labeled/scored Excel files (with `Relevance` and optional timing columns).

- `data/results_json/`
  - Raw evaluation JSON outputs (including timing breakdowns).

- `reports/`
  - Aggregated metrics artifacts (`ir_metrics_*.json/csv/xlsx`).

- Python scripts in this folder (`evaluate.py`, `get_metrics.py`, `ir_evaluation.py`) remain at the top level for stable imports and CLI usage.

## Unified Evaluation CLI

Use `evaluate.py` as the single entrypoint:

```bash
python backend/evaluation/evaluate.py run --db all
python backend/evaluation/evaluate.py run --db company
python backend/evaluation/evaluate.py run --db aiid
python backend/evaluation/evaluate.py run --db standards
```

- `--db all` runs the general multi-method evaluation.
- `--db company|aiid|standards` runs DB-restricted evaluation (with timing columns in chunk exports).

## IR Metrics Report

Generate/update the combined report:

```bash
python backend/evaluation/get_metrics.py --report --k 10
```

By default it now reads from `data/labeled_xlsx/` and writes outputs to `reports/`.

