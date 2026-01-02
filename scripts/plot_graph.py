import pandas as pd
import json
import matplotlib.pyplot as plt
from itertools import cycle
import os
import argparse
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
        # Some exports may double-quote JSON; try a second pass.
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
    # Fallback: try tab-separated then comma
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

        # Run start markers:
        # - iteration_no == 0 (most reliable for DB exports)
        # - method == BASE/base_classifier or notes == base_classifier
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


def _plot_metric(
    test_idx: int,
    test_df: pd.DataFrame,
    metric: str,
    output_dir: str,
    accuracy_percent: bool,
) -> Optional[str]:
    if "metrics_dict" not in test_df.columns:
        return None

    x_col = None
    if "iteration_no" in test_df.columns:
        x_col = "iteration_no"
    elif "train_data_size" in test_df.columns:
        x_col = "train_data_size"

    x_values = (
        test_df[x_col].tolist()
        if x_col is not None
        else list(range(len(test_df)))
    )

    y_values: List[Optional[float]] = []
    for d in test_df["metrics_dict"].tolist():
        val = None
        if isinstance(d, dict):
            val = d.get(metric)
        if metric == "accuracy" and val is not None and accuracy_percent:
            try:
                val = float(val) * 100.0
            except Exception:
                pass
        y_values.append(val)

    if all(v is None or (isinstance(v, float) and pd.isna(v)) for v in y_values):
        return None

    plt.figure(figsize=(12, 5))

    colors = cycle(["green", "blue", "purple", "orange", "brown", "gray"])
    used_labels = set()
    for i, row in test_df.iterrows():
        method = str(row.get("method", ""))
        model_name = str(row.get("model_name", ""))
        train_data_size = row.get("train_data_size", None)
        data_size = row.get("data_size", None)

        if method == "base_classifier":
            color = "red"
        else:
            color = next(colors)

        y = y_values[i] if i < len(y_values) else None
        plt.scatter(x_values[i], y, color=color, s=80, edgecolor="black", zorder=3)

        label = f"{model_name} | method={method} | train_data_size={train_data_size} | data_size={data_size}"
        if label not in used_labels:
            plt.scatter([], [], color=color, label=label)
            used_labels.add(label)

    plt.plot(x_values, y_values, color="black", linewidth=1, zorder=1)

    xlabel = x_col if x_col is not None else "Iteration"
    plt.xlabel(xlabel)

    ylabel = metric
    if metric == "accuracy" and accuracy_percent:
        ylabel = "Accuracy (%)"
    plt.ylabel(ylabel)

    stop_reason = None
    for col in ["stop_reason", "notes", "stop_condition"]:
        if col in test_df.columns:
            try:
                val = test_df[col].iloc[-1]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    stop_reason = str(val)
                    break
            except Exception:
                pass

    title = f"{metric} Plot - Graph {test_idx}"
    if stop_reason:
        title = f"{title} | stop={stop_reason}"
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)

    file_name = f"{metric}.jpeg"
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path)
    plt.close()
    return out_path


def _derive_result_folder_name(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    if base.startswith("results"):
        return "result" + base[len("results"):]
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/results.xlsx")
    parser.add_argument("--output_dir", default="graphs")
    parser.add_argument("--accuracy_percent", action="store_true")
    parser.add_argument(
        "--test_index",
        type=int,
        default=None,
        help="1-based index of the test/data block to plot (after splitting by base_classifier).",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Treat the entire input as a single test/data block (do not split by base_classifier).",
    )
    args = parser.parse_args()

    # Excel dosyasını oku
    df = _read_results(args.input)

    # metrics sütununu dict olarak al
    if "metrics" not in df.columns:
        raise KeyError("Input file must contain a 'metrics' column")
    df["metrics_dict"] = df["metrics"].apply(_safe_json_loads)

    # Kayıt dizini
    result_folder = _derive_result_folder_name(args.input)
    base_output_dir = args.output_dir
    if base_output_dir == "graphs":
        base_output_dir = os.path.join(os.path.dirname(args.input), result_folder)
    os.makedirs(base_output_dir, exist_ok=True)

    tests = [df.reset_index(drop=True)] if args.no_split else _split_into_tests(df)
    if args.test_index is not None:
        if args.test_index <= 0 or args.test_index > len(tests):
            raise ValueError(f"--test_index must be between 1 and {len(tests)} (got {args.test_index})")
        tests = [tests[args.test_index - 1]]

    for idx, test_df in enumerate(tests, 1):
        output_dir = os.path.join(base_output_dir, f"data{idx}graphs")
        os.makedirs(output_dir, exist_ok=True)

        metric_keys = set()
        for d in test_df["metrics_dict"].tolist():
            if isinstance(d, dict):
                metric_keys.update(d.keys())

        for metric in sorted(metric_keys):
            _plot_metric(
                test_idx=idx,
                test_df=test_df,
                metric=metric,
                output_dir=output_dir,
                accuracy_percent=args.accuracy_percent,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
