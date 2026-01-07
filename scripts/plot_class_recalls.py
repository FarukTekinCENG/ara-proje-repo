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

def _extract_recall_data(test_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Extract recall data for all classes from a single test dataframe"""
    if "metrics_dict" not in test_df.columns:
        return None

    # Always use iteration_no for x-axis
    x_col = "iteration_no"
    if x_col not in test_df.columns:
        return None

    x_values = test_df[x_col].tolist()

    # Find all recall metrics
    recall_metrics = []
    for d in test_df["metrics_dict"].tolist():
        if isinstance(d, dict):
            for key in d.keys():
                if key.startswith("recall_") and key not in recall_metrics:
                    recall_metrics.append(key)
    
    if not recall_metrics:
        return None

    # Prepare data for each recall metric
    recall_data = {}
    for metric in recall_metrics:
        y_values = []
        for d in test_df["metrics_dict"].tolist():
            val = None
            if isinstance(d, dict):
                val = d.get(metric)
            y_values.append(val)
        recall_data[metric] = y_values

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
        "recall_data": recall_data,
        "recall_metrics": recall_metrics,
        "method": actual_method,
        "max_train_size": max_train_size
    }

def _plot_class_recall(
    all_tests_data: List[Dict[str, Any]],
    class_name: str,
    output_dir: str,
) -> str:
    """Plot recall graph for a specific class across all tests"""
    
    plt.figure(figsize=(14, 8))
    
    # Colors for different methods
    colors = plt.cm.tab10.colors
    
    method_pretty_map = {
        "uncertainty_sampling": "Uncertainty Sampling",
        "diversity_sampling": "Diversity Sampling",
        "query_by_comitee": "Query by Committee",
        "random_sampling": "Random Sampling",
    }
    
    # Plot each method for this specific class
    for i, test_data in enumerate(all_tests_data):
        if test_data is None or "recall_data" not in test_data:
            continue
            
        method_name = test_data["method"]
        method_pretty = method_pretty_map.get(method_name, method_name)
        
        # Check if this is the last test and method is random_sampling
        is_last_test = (i == len(all_tests_data) - 1)
        if is_last_test and method_name == "random_sampling":
            max_train_size = test_data.get("max_train_size", 0)
            if max_train_size and max_train_size >= 10000:
                method_pretty = "Full Size Labelling"
        
        x_values = test_data["x_values"]
        recall_data = test_data["recall_data"]
        
        # Find the recall metric for this class
        class_metric = f"recall_{class_name.lower().replace(' ', '_')}"
        if class_metric in recall_data:
            y_values = recall_data[class_metric]
            
            # Plot with method-specific color
            color = colors[i % len(colors)]
            plt.plot(x_values, y_values, 
                    color=color, 
                    linewidth=2.5, 
                    marker='o', 
                    markersize=6,
                    markeredgecolor='black',
                    markerfacecolor=color,
                    markeredgewidth=1.5,
                    label=method_pretty,
                    alpha=0.8)
    
    # Formatting
    plt.xlabel("Iteration", fontsize=12, fontweight='bold')
    plt.ylabel("Recall", fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    
    # Set x-axis to show every iteration
    max_iteration = max(max(test_data["x_values"]) for test_data in all_tests_data if test_data)
    plt.xticks(range(0, max_iteration + 1, 1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    
    # Title
    title = f"Combined {class_name.title()} Class Recall Performance Across All Tests"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Layout
    plt.tight_layout()
    
    # Save
    file_name = f"combined_recall_{class_name.lower().replace(' ', '_')}.jpeg"
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return out_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Create combined recall graphs for each class")
    parser.add_argument("--input", default="results/results_final/results_N3000.csv")
    parser.add_argument("--output_dir", default="results/results_final/result_N3000/combined_graphs")
    
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
    
    # Extract recall data from all tests
    recall_tests_data = []
    for test_df in tests:
        test_data = _extract_recall_data(test_df)
        if test_data:
            recall_tests_data.append(test_data)
    
    if not recall_tests_data:
        print("No valid recall data found")
        return 1
    
    # Collect all unique class names across all tests
    all_classes = set()
    for test_data in recall_tests_data:
        if test_data and "recall_metrics" in test_data:
            for metric in test_data["recall_metrics"]:
                class_name = metric.replace("recall_", "").replace("_", " ").title()
                all_classes.add(class_name)
    
    all_classes = sorted(list(all_classes))
    
    print(f"Found {len(all_classes)} classes: {', '.join(all_classes)}")
    
    # Create a graph for each class
    for class_name in all_classes:
        print(f"Processing {class_name}...")
        
        output_path = _plot_class_recall(
            all_tests_data=recall_tests_data,
            class_name=class_name,
            output_dir=args.output_dir
        )
        
        print(f"Combined {class_name} recall graph saved to: {output_path}")
    
    print(f"All class recall graphs saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
