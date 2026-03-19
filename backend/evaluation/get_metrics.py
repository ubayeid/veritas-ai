"""
Calculate and display accuracy and timing metrics per method from evaluation results.
"""

import json
import csv
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)


def _load_labels_table(labels_file: str) -> "pd.DataFrame":
    """Load labeled table from CSV or Excel."""
    path = Path(labels_file)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(labels_file)
    return pd.read_csv(labels_file)


def calculate_accuracy_metrics(labels_file: str, k: int = 10) -> Dict[str, Dict]:
    """Calculate accuracy metrics per method from labeled CSV/XLSX."""
    df = _load_labels_table(labels_file)
    
    if 'Relevance' not in df.columns:
        print("Warning: CSV file does not have 'Relevance' column. Accuracy metrics cannot be calculated.")
        return {}
    
    methods = ['VECTOR', 'GRAPH', 'HYBRID']
    all_method_metrics = {}
    
    for method in methods:
        method_df = df[df['Method'] == method].copy()
        if method_df.empty:
            continue
        
        per_query_metrics = []
        for query_id in method_df['Query ID'].unique():
            query_df = method_df[method_df['Query ID'] == query_id]
            query_df = query_df.sort_values('Rank').head(k)
            
            # Calculate metrics
            relevant_count = (query_df['Relevance'] >= 1).sum()
            total_relevant = (df[df['Query ID'] == query_id]['Relevance'] >= 1).sum()
            
            precision = relevant_count / k if k > 0 else 0.0
            recall = relevant_count / total_relevant if total_relevant > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # MRR
            mrr = 0.0
            for rank, rel in enumerate(query_df['Relevance'], start=1):
                if rel >= 1:
                    mrr = 1.0 / rank
                    break
            
            # MAP
            map_score = 0.0
            relevant_so_far = 0
            precision_sum = 0.0
            for rank, rel in enumerate(query_df['Relevance'], start=1):
                if rel >= 1:
                    relevant_so_far += 1
                    precision_sum += relevant_so_far / rank
            if total_relevant > 0:
                map_score = precision_sum / total_relevant
            
            # NDCG
            import math
            dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(query_df['Relevance'], start=1))
            ideal_relevances = sorted(df[df['Query ID'] == query_id]['Relevance'], reverse=True)[:k]
            idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            per_query_metrics.append({
                'query_id': query_id,
                'precision_at_k': precision,
                'recall_at_k': recall,
                'f1_at_k': f1,
                'mrr': mrr,
                'map': map_score,
                'ndcg_at_k': ndcg
            })
        
        # Aggregate
        if per_query_metrics:
            aggregate = {
                'avg_precision_at_k': sum(m['precision_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                'avg_recall_at_k': sum(m['recall_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                'avg_f1_at_k': sum(m['f1_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                'avg_mrr': sum(m['mrr'] for m in per_query_metrics) / len(per_query_metrics),
                'avg_map': sum(m['map'] for m in per_query_metrics) / len(per_query_metrics),
                'avg_ndcg_at_k': sum(m['ndcg_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                'total_queries': len(per_query_metrics)
            }
            
            all_method_metrics[method.lower()] = {
                'per_query': per_query_metrics,
                'aggregate': aggregate
            }
    
    return all_method_metrics


def calculate_timing_metrics(results_file: str) -> Dict[str, Dict]:
    """Calculate timing metrics per method from evaluation results JSON."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    timing_by_method = defaultdict(list)
    
    for result in results:
        timing = result.get('timing_by_method', {})
        for method, time_sec in timing.items():
            timing_by_method[method].append(time_sec)
    
    # Calculate aggregate timing metrics
    timing_metrics = {}
    for method, times in timing_by_method.items():
        if times:
            timing_metrics[method] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times),
                'total_queries': len(times)
            }
    
    return timing_metrics


def get_timing_from_csv(csv_file: str) -> Dict[str, Dict]:
    """Extract timing metrics from CSV if timing columns exist."""
    df = pd.read_csv(csv_file)
    
    # Check if timing columns exist
    timing_columns = [col for col in df.columns if 'timing' in col.lower() or 'time' in col.lower()]
    if not timing_columns:
        return {}
    
    # Group by method and calculate timing
    timing_by_method = defaultdict(list)
    
    for method in ['VECTOR', 'GRAPH', 'HYBRID']:
        method_df = df[df['Method'] == method]
        if method_df.empty:
            continue
        
        # Try to find timing column for this method
        method_timing_col = f'{method}_Time'
        if method_timing_col in df.columns:
            times = method_df[method_timing_col].dropna().tolist()
            timing_by_method[method.lower()].extend(times)
    
    # Calculate aggregate timing metrics
    timing_metrics = {}
    for method, times in timing_by_method.items():
        if times:
            timing_metrics[method] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times),
                'total_queries': len(times)
            }
    
    return timing_metrics


def display_metrics(accuracy_metrics: Dict, timing_metrics: Dict, k: int = 10):
    """Display accuracy and timing metrics in a formatted table."""
    print("\n" + "=" * 100)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 100)
    
    # Display accuracy metrics
    if accuracy_metrics:
        print("\n" + "-" * 100)
        print("ACCURACY METRICS (per method)")
        print("-" * 100)
        print(f"\n{'Method':<15} {'P@{k}':<10} {'R@{k}':<10} {'F1@{k}':<10} {'MRR':<10} {'MAP':<10} {'NDCG@{k}':<10} {'Queries':<10}")
        print("-" * 100)
        
        for method in ['vector', 'graph', 'hybrid']:
            if method in accuracy_metrics:
                agg = accuracy_metrics[method]['aggregate']
                print(f"{method.capitalize():<15} "
                      f"{agg['avg_precision_at_k']:<10.3f} "
                      f"{agg['avg_recall_at_k']:<10.3f} "
                      f"{agg['avg_f1_at_k']:<10.3f} "
                      f"{agg['avg_mrr']:<10.3f} "
                      f"{agg['avg_map']:<10.3f} "
                      f"{agg['avg_ndcg_at_k']:<10.3f} "
                      f"{agg['total_queries']:<10}")
    else:
        print("\n" + "-" * 100)
        print("ACCURACY METRICS: Not available (CSV file needs 'Relevance' column for labeling)")
        print("-" * 100)
    
    # Display timing metrics
    if timing_metrics:
        print("\n" + "-" * 100)
        print("TIMING METRICS (per method)")
        print("-" * 100)
        print(f"\n{'Method':<15} {'Avg Time (s)':<15} {'Min Time (s)':<15} {'Max Time (s)':<15} {'Total Time (s)':<15} {'Queries':<10}")
        print("-" * 100)
        
        for method in ['vector', 'graph', 'hybrid']:
            if method in timing_metrics:
                tm = timing_metrics[method]
                print(f"{method.capitalize():<15} "
                      f"{tm['avg_time']:<15.3f} "
                      f"{tm['min_time']:<15.3f} "
                      f"{tm['max_time']:<15.3f} "
                      f"{tm['total_time']:<15.3f} "
                      f"{tm['total_queries']:<10}")
    else:
        print("\n" + "-" * 100)
        print("TIMING METRICS: Not available")
        print("-" * 100)
    
    print("\n" + "=" * 100)


def _print_how_to_get_metrics():
    """Print instructions when no metrics are available."""
    print("\n" + "=" * 100)
    print("HOW TO GET METRICS:")
    print("=" * 100)
    print("\n1. For ACCURACY METRICS:")
    print("   - Add a 'Relevance' column to your CSV file")
    print("   - Label each chunk: 0=not relevant, 1=partially relevant, 2=highly relevant")
    print("   - Run: python backend/evaluation/get_metrics.py --labels your_file.csv")
    print("\n2. For TIMING METRICS:")
    print("   - Re-run evaluation to capture timing:")
    print("     python backend/evaluation/evaluate.py run --queries <queries_file> --output <output.csv>")
    print("   - This will create a <output>_results.json file with timing data")
    print("   - Then run: python backend/evaluation/get_metrics.py --results <output>_results.json")
    print("\n3. For BOTH:")
    print("   - Run evaluation with timing, then label the CSV, then run this script with both files")


METHOD_LABEL_TO_KEY = {
    "VECTOR": "vector",
    "GRAPH": "graph",
    "HYBRID": "hybrid",
}


def _extract_timing_table_from_df(df: "pd.DataFrame") -> "pd.DataFrame":
    """Extract one timing row per (Query ID, Method), averaged across ranks."""
    timing_cols = ["total_s", "retrieval_s", "answer_s", "non_answer_s"]
    present = [c for c in timing_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    if "Query ID" not in df.columns or "Method" not in df.columns:
        return pd.DataFrame()

    reduced = (
        df.groupby(["Query ID", "Method"], dropna=False)[present]
        .mean(numeric_only=True)
        .reset_index()
    )
    reduced["method"] = reduced["Method"].map(METHOD_LABEL_TO_KEY)
    reduced = reduced.dropna(subset=["method"])
    return reduced[["Query ID", "method"] + present]


def _compute_effective_k(df: "pd.DataFrame", requested_k: int) -> int:
    if "Rank" not in df.columns:
        return requested_k
    try:
        max_rank = int(pd.to_numeric(df["Rank"], errors="coerce").max())
    except Exception:
        return requested_k
    if max_rank <= 0:
        return requested_k
    return min(requested_k, max_rank)


def _paired_permutation_pvalue(
    values_a: List[float],
    values_b: List[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired, two-sided permutation test using sign-flips on pairwise differences.

    Returns:
      - mean_a, mean_b, mean_diff
      - p_value (two-sided)
      - n_pairs
    """
    n = min(len(values_a), len(values_b))
    if n == 0:
        return {"mean_a": float("nan"), "mean_b": float("nan"), "mean_diff": float("nan"), "p_value": float("nan"), "n_pairs": 0}

    diffs = [float(a) - float(b) for a, b in zip(values_a[:n], values_b[:n])]
    observed = sum(diffs) / n
    obs_abs = abs(observed)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_permutations):
        perm_mean = sum((d if rng.random() < 0.5 else -d) for d in diffs) / n
        if abs(perm_mean) >= obs_abs:
            extreme += 1

    # add-one smoothing
    p_value = (extreme + 1) / (n_permutations + 1)
    return {
        "mean_a": sum(values_a[:n]) / n,
        "mean_b": sum(values_b[:n]) / n,
        "mean_diff": observed,
        "p_value": p_value,
        "n_pairs": n,
    }


def _compute_significance_tests_for_dataset(
    dataset_key: str,
    accuracy_metrics: Dict[str, Dict[str, Any]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Compute paired permutation significance tests across method pairs.
    Uses only query_ids common to both methods in each pair.
    """
    method_pairs = [("vector", "graph"), ("vector", "hybrid"), ("graph", "hybrid")]
    metric_keys = ["precision_at_k", "recall_at_k", "f1_at_k", "mrr", "map", "ndcg_at_k"]

    per_method_q = {}
    for method in ["vector", "graph", "hybrid"]:
        if method not in accuracy_metrics:
            continue
        rows = accuracy_metrics[method].get("per_query", []) or []
        per_method_q[method] = {r["query_id"]: r for r in rows if "query_id" in r}

    rows_out: List[Dict[str, Any]] = []
    for method_a, method_b in method_pairs:
        if method_a not in per_method_q or method_b not in per_method_q:
            continue
        common_qids = sorted(set(per_method_q[method_a].keys()) & set(per_method_q[method_b].keys()))
        if not common_qids:
            continue

        for metric in metric_keys:
            a_vals = []
            b_vals = []
            for qid in common_qids:
                a = per_method_q[method_a][qid].get(metric)
                b = per_method_q[method_b][qid].get(metric)
                if a is None or b is None:
                    continue
                a_vals.append(float(a))
                b_vals.append(float(b))

            res = _paired_permutation_pvalue(
                a_vals,
                b_vals,
                n_permutations=n_permutations,
                seed=seed,
            )
            rows_out.append(
                {
                    "dataset": dataset_key,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_pairs": res["n_pairs"],
                    "mean_a": res["mean_a"],
                    "mean_b": res["mean_b"],
                    "mean_diff_a_minus_b": res["mean_diff"],
                    "p_value_two_sided": res["p_value"],
                    "n_permutations": n_permutations,
                }
            )

    return rows_out


def _bh_adjust(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR adjustment.
    Returns q-values in the original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = list(enumerate(p_values))
    indexed.sort(key=lambda x: x[1])

    # Raw adjusted values
    adjusted = [0.0] * m
    for rank, (_, p) in enumerate(indexed, start=1):
        adjusted[rank - 1] = p * m / rank

    # Enforce monotonicity from largest rank to smallest
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Map back to original order
    out = [0.0] * m
    for sorted_pos, (orig_idx, _) in enumerate(indexed):
        out[orig_idx] = min(1.0, adjusted[sorted_pos])
    return out


def _apply_bh_correction(rows: List[Dict[str, Any]], alpha: float = 0.05) -> List[Dict[str, Any]]:
    """
    Add BH-corrected q-values and significance flags.
    Applies correction:
      1) globally across all tests
      2) within each dataset
    """
    if not rows:
        return rows

    # Global correction
    p_global = [float(r["p_value_two_sided"]) for r in rows]
    q_global = _bh_adjust(p_global)
    for r, q in zip(rows, q_global):
        r["q_value_bh_global"] = q
        r[f"significant_bh_global_alpha_{str(alpha).replace('.', '_')}"] = bool(q < alpha)

    # Dataset-level correction
    by_dataset: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        by_dataset[str(r.get("dataset", ""))].append(idx)

    for ds, idxs in by_dataset.items():
        pvals = [float(rows[i]["p_value_two_sided"]) for i in idxs]
        qvals = _bh_adjust(pvals)
        for i, q in zip(idxs, qvals):
            rows[i]["q_value_bh_dataset"] = q
            rows[i][f"significant_bh_dataset_alpha_{str(alpha).replace('.', '_')}"] = bool(q < alpha)

    return rows


def generate_ir_metrics_report(
    k: int,
    out_dir: Path,
    datasets: Dict[str, Path],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, str]:
    """Generate consolidated multi-dataset IR report (json + xlsx)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "k_requested": k,
        "significance_test": {
            "name": "paired_two_sided_permutation_sign_flip",
            "n_permutations": n_permutations,
            "seed": seed,
            "multiple_testing_correction": {
                "method": "benjamini_hochberg_fdr",
                "alpha": 0.05,
            },
        },
        "datasets": {},
        "comparison_table": [],
        "significance_tests": [],
    }
    timing_by_query_rows: List[Dict[str, Any]] = []

    for dataset_key, file_path in datasets.items():
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        df = pd.read_excel(file_path)
        k_used = _compute_effective_k(df, k)
        accuracy_metrics = calculate_accuracy_metrics(str(file_path), k=k_used)

        timing_table = _extract_timing_table_from_df(df)
        timing_metrics: Dict[str, Dict[str, Any]] = {}
        if not timing_table.empty:
            for method_key in ["vector", "graph", "hybrid"]:
                sub = timing_table[timing_table["method"] == method_key].copy()
                if sub.empty:
                    continue
                m: Dict[str, Any] = {"total_queries": int(sub["Query ID"].nunique(dropna=True))}
                for col in ["total_s", "retrieval_s", "answer_s", "non_answer_s"]:
                    series = pd.to_numeric(sub[col], errors="coerce").dropna()
                    if not series.empty:
                        m[f"avg_{col}"] = float(series.mean())
                timing_metrics[method_key] = m

            for _, r in timing_table.iterrows():
                timing_by_query_rows.append(
                    {
                        "dataset": dataset_key,
                        "query_id": r["Query ID"],
                        "method": r["method"],
                        "total_s": float(r["total_s"]),
                        "retrieval_s": float(r["retrieval_s"]),
                        "answer_s": float(r["answer_s"]),
                        "non_answer_s": float(r["non_answer_s"]),
                    }
                )

        report["datasets"][dataset_key] = {
            "file": str(file_path),
            "k_used": k_used,
            "accuracy_metrics": accuracy_metrics,
            "timing_metrics": timing_metrics,
        }

        dataset_sig = _compute_significance_tests_for_dataset(
            dataset_key=dataset_key,
            accuracy_metrics=accuracy_metrics,
            n_permutations=n_permutations,
            seed=seed,
        )
        report["datasets"][dataset_key]["significance_tests"] = dataset_sig
        report["significance_tests"].extend(dataset_sig)

        for method_key in ["vector", "graph", "hybrid"]:
            if method_key not in accuracy_metrics:
                continue
            agg = accuracy_metrics[method_key]["aggregate"]
            timing = timing_metrics.get(method_key, {})
            row = {
                "dataset": dataset_key,
                "method": method_key,
                "k_used": k_used,
                "total_queries": agg.get("total_queries"),
                "avg_precision_at_k": agg.get("avg_precision_at_k"),
                "avg_recall_at_k": agg.get("avg_recall_at_k"),
                "avg_f1_at_k": agg.get("avg_f1_at_k"),
                "avg_mrr": agg.get("avg_mrr"),
                "avg_map": agg.get("avg_map"),
                "avg_ndcg_at_k": agg.get("avg_ndcg_at_k"),
            }
            for timing_col in ["total_s", "retrieval_s", "answer_s", "non_answer_s"]:
                row[f"avg_{timing_col}"] = timing.get(f"avg_{timing_col}")
            report["comparison_table"].append(row)

    json_path = out_dir / "ir_metrics_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_df = pd.DataFrame(report["comparison_table"])

    sig_df = pd.DataFrame(report["significance_tests"])
    if not sig_df.empty:
        # Keep JSON and tabular outputs aligned with corrected values/flags.
        corrected_rows = _apply_bh_correction(report["significance_tests"], alpha=0.05)
        report["significance_tests"] = corrected_rows
        # Also update nested dataset copies
        by_dataset_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in corrected_rows:
            by_dataset_rows[str(r["dataset"])].append(r)
        for ds_key, rows in by_dataset_rows.items():
            if ds_key in report["datasets"]:
                report["datasets"][ds_key]["significance_tests"] = rows
        sig_df = pd.DataFrame(corrected_rows)

    # Re-write JSON once more so correction fields are persisted.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    xlsx_path = out_dir / "ir_metrics_report.xlsx"
    xlsx_written_path = xlsx_path
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            sig_df.to_excel(writer, sheet_name="significance_tests", index=False)
            if timing_by_query_rows:
                pd.DataFrame(timing_by_query_rows).to_excel(writer, sheet_name="timing_by_query", index=False)
    except PermissionError:
        import datetime as _dt
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_written_path = out_dir / f"ir_metrics_report_{stamp}.xlsx"
        with pd.ExcelWriter(xlsx_written_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            sig_df.to_excel(writer, sheet_name="significance_tests", index=False)
            if timing_by_query_rows:
                pd.DataFrame(timing_by_query_rows).to_excel(writer, sheet_name="timing_by_query", index=False)

    return {
        "json_path": str(json_path),
        "xlsx_path": str(xlsx_written_path),
    }


def main():
    """Main entry point. Shows metrics with sensible defaults and auto-discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate and display accuracy and timing metrics per method. "
        "Works with labeled CSV files and/or JSON results files."
    )
    parser.add_argument(
        '--labels', type=str,
        default='backend/evaluation/retrieval_results.csv',
        help='Labeled CSV file with Relevance column (default: backend/evaluation/retrieval_results.csv)'
    )
    parser.add_argument(
        '--results', type=str,
        help='JSON file with evaluation results (for timing). Auto-discovered from labels if not given.'
    )
    parser.add_argument(
        '--csv', type=str,
        help='CSV file (can contain both labels and timing) - overrides --labels if both given'
    )
    parser.add_argument('--k', type=int, default=10, help='Top K for metrics (default: 10)')
    parser.add_argument('--output', type=str, help='Output JSON file for metrics')
    parser.add_argument('--report', action='store_true', help='Generate consolidated multi-dataset IR report')
    parser.add_argument('--report-out-dir', type=str, default='backend/evaluation/reports',
                        help='Output dir for consolidated report files')
    parser.add_argument('--report-data-dir', type=str, default='backend/evaluation/data/labeled_xlsx',
                        help='Directory containing *_scored_with_relevance.xlsx files')
    parser.add_argument('--report-permutations', type=int, default=10000,
                        help='Permutation count for significance tests (default: 10000)')
    parser.add_argument('--report-seed', type=int, default=42,
                        help='Random seed for significance tests (default: 42)')
    
    args = parser.parse_args()

    if args.report:
        data_dir = Path(args.report_data_dir)
        if not data_dir.is_absolute():
            data_dir = project_root / args.report_data_dir
        out_dir = Path(args.report_out_dir)
        if not out_dir.is_absolute():
            out_dir = project_root / args.report_out_dir

        datasets = {
            "combined": data_dir / "chunks_for_labeling_combined_scored_with_relevance.xlsx",
            "aiid": data_dir / "chunks_for_labeling_aiid_scored_with_relevance.xlsx",
            "std": data_dir / "chunks_for_labeling_standards_scored_with_relevance.xlsx",
            "company": data_dir / "chunks_for_labeling_company_scored_with_relevance.xlsx",
        }
        paths = generate_ir_metrics_report(
            k=args.k,
            out_dir=out_dir,
            datasets=datasets,
            n_permutations=args.report_permutations,
            seed=args.report_seed,
        )
        print("IR metrics report generated:")
        print(f"- {paths['json_path']}")
        print(f"- {paths['xlsx_path']}")
        return
    
    accuracy_metrics = {}
    timing_metrics = {}
    
    # Resolve labels/csv source (relative paths from project root)
    labels_file = args.csv or args.labels
    labels_path = Path(labels_file)
    if not labels_path.is_absolute():
        labels_path = project_root / labels_file
    
    # Calculate accuracy metrics
    if labels_path.exists():
        print(f"Checking {labels_path} for accuracy metrics...")
        try:
            accuracy_metrics = calculate_accuracy_metrics(str(labels_path), k=args.k)
        except Exception as e:
            print(f"  Could not calculate accuracy metrics: {e}")
    
    # Calculate timing metrics
    if args.results:
        results_path = Path(args.results)
        if results_path.exists():
            print(f"Loading timing metrics from {results_path}...")
            try:
                timing_metrics = calculate_timing_metrics(str(results_path))
            except Exception as e:
                print(f"  Could not calculate timing metrics: {e}")
    elif args.csv:
        try:
            timing_metrics = get_timing_from_csv(args.csv)
        except Exception as e:
            print(f"  Could not extract timing from CSV: {e}")
    else:
        # Auto-discover from labels: <stem>_results.json
        results_path = labels_path.parent / f"{labels_path.stem}_results.json"
        if results_path.exists():
            print(f"Found results file: {results_path}")
            try:
                timing_metrics = calculate_timing_metrics(str(results_path))
            except Exception as e:
                print(f"  Could not calculate timing metrics: {e}")
    
    # Display metrics
    display_metrics(accuracy_metrics, timing_metrics, k=args.k)
    
    # Instructions if no metrics available
    if not accuracy_metrics and not timing_metrics:
        _print_how_to_get_metrics()
    
    # Save to JSON if output file specified
    if args.output:
        output_data = {
            "k": args.k,
            "accuracy_metrics": accuracy_metrics,
            "timing_metrics": timing_metrics
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
