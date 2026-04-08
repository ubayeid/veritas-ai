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
- `--db company|aiid|standards` runs DB-restricted evaluation.

Chunk CSV exports from **`run ... --export-chunks`** (including `--db all`) include the same timing columns (`total_s`, `retrieval_s`, `answer_s`, `non_answer_s`) as single-DB runs.

If a scored Excel file has no timing columns, **`get_metrics.py --report`** loads timing from **`data/results_json/<same_stem>_results.json`** when present (see `timing_fallback` in the report JSON). Older combined JSON may only have retrieval times (`timing_by_method`); re-run evaluation for full breakdown.

**Combined (top‑8) manual scoring:** use  
`data/labeling_csv/chunks_for_labeling_combined_top8_for_scoring.csv`  
(with `…_results.json` for timing). When scored, save Excel as  
`data/labeled_xlsx/chunks_for_labeling_combined_top8_for_scoring_scored_with_relevance.xlsx`  
so **`get_metrics.py --report`** picks it up.

Optional: copy timing into the scored workbook (keeps **Relevance** labels), then regenerate the report:

```bash
python backend/evaluation/get_metrics.py --merge-timing-xlsx backend/evaluation/data/labeled_xlsx/chunks_for_labeling_combined_top8_for_scoring_scored_with_relevance.xlsx
python backend/evaluation/get_metrics.py --report --k 8
```

Use the same **K** as `evaluate.py --top-k` (e.g. 8).

Optional: **`--merge-timing-results path/to/custom_results.json`** if the JSON name does not follow the default pattern.

### Same K for retrieval, answers, and metrics

Use one value everywhere:

- **`evaluate.py run ... --top-k K`** — how many chunks per method are retrieved; answer generation uses the same cap (`top_n=K`, up to the number of chunks returned).
- **`get_metrics.py ... --k K`** (and **`--report --k K`**) — Precision/Recall/F1/NDCG cutoffs must match that same **K** so the benchmark is consistent.

Timing in saved JSON is aligned across `--db all` and `--db company|aiid|standards`:

- **`timing_by_method`**: retrieval-only seconds per method (same meaning as in `IREvaluator.evaluate_query`).
- **`timing_breakdown_by_method`**: `total_s`, `retrieval_s`, `answer_s`, `non_answer_s` where `non_answer_s = total_s - answer_s`.

Wall-clock times still differ between `--db all` and a single-DB run (different corpora and hybrid implementation), but the **definitions** of these fields match.

## IR Metrics Report

Generate/update the combined report:

```bash
python backend/evaluation/get_metrics.py --report --k 8
```

Use the same **K** as in `evaluate.py --top-k`. By default it reads from `data/labeled_xlsx/` and writes outputs to `reports/`.

