import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.ticker import MultipleLocator
from typing import Any, Dict, List, Optional

def _safe_json_loads(x: Any) -> Dict[str, Any]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return {}
    if isinstance(x, dict):
        return x
    s = str(x)
    if not s or s.lower() == "nan":
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("''", "'"))
        except Exception:
            return {}

def _read_results(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext in {".csv"}:
        return pd.read_csv(path)
    if ext in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)

def _split_into_tests(df: pd.DataFrame) -> List[pd.DataFrame]:
    tests: List[pd.DataFrame] = []
    current_rows: List[int] = []

    method_col = "method" if "method" in df.columns else None
    iter_col = "iteration_no" if "iteration_no" in df.columns else None
    notes_col = "notes" if "notes" in df.columns else None

    def _is_run_start(r: pd.Series) -> bool:
        m = str(r.get(method_col, "")) if method_col else ""
        n = str(r.get(notes_col, "")) if notes_col else ""
        it0 = False
        if iter_col:
            try:
                it0 = int(r.get(iter_col)) == 0
            except Exception:
                it0 = False

        if it0:
            return True
        if m.upper() == "BASE" or m == "base_classifier":
            return True
        if n == "base_classifier":
            return True
        return False

    for idx, row in df.iterrows():
        if _is_run_start(row) and current_rows:
            tests.append(df.loc[current_rows].reset_index(drop=True))
            current_rows = [idx]
        else:
            if not current_rows:
                current_rows = [idx]
            else:
                current_rows.append(idx)

    if current_rows:
        tests.append(df.loc[current_rows].reset_index(drop=True))

    return tests

def _extract_metric_data(test_df: pd.DataFrame, metric: str) -> Optional[Dict[str, Any]]:
    """Extract metric data from a single test dataframe"""
    if "metrics_dict" not in test_df.columns:
        return None

    # Always use iteration_no for x-axis
    x_col = "iteration_no"
    if x_col not in test_df.columns:
        return None

    x_values = test_df[x_col].tolist()

    y_values = []
    for d in test_df["metrics_dict"].tolist():
        val = None
        if isinstance(d, dict):
            val = d.get(metric)
        y_values.append(val)

    if all(v is None or (isinstance(v, float) and pd.isna(v)) for v in y_values):
        return None

    # Get max train_data_size to detect full dataset usage
    max_train_size = None
    if "train_data_size" in test_df.columns:
        try:
            max_train_size = max(test_df["train_data_size"])
        except Exception:
            max_train_size = None

    # Get the actual method name (skip BASE rows, find the real method)
    actual_method = "unknown"
    if "method" in test_df.columns:
        methods = test_df["method"].tolist()
        for method in methods:
            if method and method.upper() != "BASE" and method.strip():
                actual_method = method.strip()
                break
    
    return {
        "x_values": x_values,
        "y_values": y_values,
        "method": actual_method,
        "max_train_size": max_train_size
    }

def _plot_combined_metric(
    all_tests_data: List[Dict[str, Any]],
    metric: str,
    output_dir: str,
    accuracy_percent: bool = False
) -> str:
    """Plot combined metric graph from all tests"""
    
    plt.figure(figsize=(14, 8))
    
    # Colors for different tests
    colors = plt.cm.tab10.colors
    
    method_pretty_map = {
        "uncertainty_sampling": "Uncertainty Sampling",
        "diversity_sampling": "Diversity Sampling",
        "query_by_comitee": "Query by Committee",
        "random_sampling": "Random Sampling",
    }
    
    metric_pretty_map = {
        "accuracy": "Accuracy",
        "macro_f1": "F1-Score",
        "avg_uncertainty_pool": "Average Uncertainty of Pool",
        "selected_samples_avg_uncertainty": "Average Uncertainty of Selected Samples",
    }
    
    metric_pretty = metric_pretty_map.get(metric, metric)
    
    for i, test_data in enumerate(all_tests_data):
        if test_data is None:
            continue
            
        x_values = test_data["x_values"]
        y_values = test_data["y_values"]
        method_name = test_data["method"]
        
        # Convert accuracy to percentage if needed
        if metric == "accuracy" and accuracy_percent:
            y_values = [v * 100 if v is not None else None for v in y_values]
        
        # Get method name and check if it's the last test with random sampling
        method_pretty = method_pretty_map.get(method_name, method_name)
        
        # Check if this is the last test and method is random_sampling
        is_last_test = (i == len(all_tests_data) - 1)
        if is_last_test and method_name == "random_sampling":
            # Check if this test uses full dataset by looking at max_train_size
            max_train_size = test_data.get("max_train_size", 0)
            if max_train_size and max_train_size >= 10000:  # 10000 means full dataset
                method_pretty = "Full Size Labelling"
        
        label = method_pretty
        
        # Plot with markers
        plt.plot(x_values, y_values, 
                color=colors[i % len(colors)], 
                linewidth=2.5, 
                marker='o', 
                markersize=6, 
                markeredgecolor='black',
                markerfacecolor=colors[i % len(colors)],
                markeredgewidth=1.5,
                label=label,
                alpha=0.8)
    
    # Formatting
    plt.xlabel("Iteration", fontsize=12, fontweight='bold')
    
    ylabel = metric_pretty
    if metric == "accuracy" and accuracy_percent:
        ylabel = "Accuracy (%)"
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # Set x-axis to show every iteration (1, 2, 3, ...)
    max_iteration = max(max(test_data["x_values"]) for test_data in all_tests_data if test_data)
    plt.xticks(range(0, max_iteration + 1, 1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    
    # Set y-axis limits and scale
    if metric == "accuracy" and accuracy_percent:
        plt.ylim(25, 100)  # Wide range to include BASE (28.6) and all iterations
        plt.gca().yaxis.set_major_locator(MultipleLocator(10))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
    elif metric == "avg_uncertainty_pool":
        # Linear scale with wide range for better differentiation
        plt.ylim(0.1, 0.9)  # Wide range to show all uncertainty data
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
    elif "selected_samples_avg_uncertainty" in metric:
        # Linear scale with wide range for better differentiation
        plt.yscale('linear')  # Explicitly linear scale
        plt.ylim(0.0, 1.0)  # Full range from 0 to 1
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
    else:
        # Linear scale with WIDE range for better differentiation
        # Explicitly set linear scale to ensure no log scale is applied
        plt.yscale('linear')  # Explicitly linear scale
        # Set appropriate limits based on metric
        if metric == "macro_f1":
            plt.ylim(0.1, 0.85)  # VERY WIDE range - from low values to high values
        elif "recall" in metric:
            plt.ylim(0.0, 1.0)  # Full range for recall
        else:
            plt.ylim(0.1, 1.0)  # Default wide range
        
        # Fine-grained linear scale tick marks for better differentiation
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
    
    # Title
    title = f"Combined {metric_pretty} Performance Across All Tests"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    plt.legend(loc="best", fontsize=20, framealpha=0.9)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Layout
    plt.tight_layout()
    
    # Save
    file_name = f"combined_{metric}.jpeg"
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Create combined metric graphs across all tests")
    parser.add_argument("--input", default="results/results_final/results_N3000.csv")
    parser.add_argument("--output_dir", default="results/results_final/result_N3000/combined_graphs")
    parser.add_argument("--accuracy_percent", action="store_true", help="Show accuracy as percentage")
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "macro_f1", "avg_uncertainty_pool", "selected_samples_avg_uncertainty"], 
                       help="Metrics to combine (default: accuracy macro_f1 avg_uncertainty_pool selected_samples_avg_uncertainty)")
    
    args = parser.parse_args()
    
    # Read input file
    df = _read_results(args.input)
    
    # Convert metrics column to dict
    if "metrics" not in df.columns:
        raise KeyError("Input file must contain a 'metrics' column")
    df["metrics_dict"] = df["metrics"].apply(_safe_json_loads)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split into tests
    tests = _split_into_tests(df)
    
    # Process each metric
    for metric in args.metrics:
        print(f"Processing {metric}...")
        
        # Extract data from all tests
        all_tests_data = []
        for test_df in tests:
            test_data = _extract_metric_data(test_df, metric)
            if test_data:
                all_tests_data.append(test_data)
        
        if not all_tests_data:
            print(f"No valid data found for metric: {metric}")
            continue
        
        # Plot combined graph
        output_path = _plot_combined_metric(
            all_tests_data=all_tests_data,
            metric=metric,
            output_dir=args.output_dir,
            accuracy_percent=args.accuracy_percent
        )
        
        print(f"Combined {metric} graph saved to: {output_path}")
    
    print(f"All combined graphs saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
