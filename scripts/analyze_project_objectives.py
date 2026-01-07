#!/usr/bin/env python3
"""
Aktif Öğrenme Analiz Scripti
Proje hedeflerine göre otomatik analiz yapar
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple, Any

def _safe_json_loads(x: Any) -> Dict[str, Any]:
    """Safe JSON loading with error handling"""
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
    """Read results from Excel/CSV/TSV files"""
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
    """Split dataframe into test blocks using same logic as plot_graph.py"""
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

def analyze_all_tests_combined(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze all test blocks combined for project objectives"""
    
    # Extract macro_f1 and accuracy from metrics
    df["metrics_dict"] = df["metrics"].apply(_safe_json_loads)
    macro_f1_values = []
    accuracy_values = []
    for metrics_dict in df["metrics_dict"]:
        if isinstance(metrics_dict, dict):
            macro_f1_values.append(metrics_dict.get("macro_f1", np.nan))
            accuracy_values.append(metrics_dict.get("accuracy", np.nan))
        else:
            macro_f1_values.append(np.nan)
            accuracy_values.append(np.nan)
    df["macro_f1"] = macro_f1_values
    df["accuracy"] = accuracy_values
    
    # Filter out base classifier
    valid_df = df[~df['method'].isin(['base_classifier', 'BASE'])].copy()
    
    if len(valid_df) == 0:
        return {"error": "No valid data found"}
    
    # Group by method and get final results
    final_results = {}
    for method in valid_df['method'].unique():
        method_data = valid_df[valid_df['method'] == method]
        if len(method_data) > 0:
            # Get the best result for this method (highest combined score)
            # We'll use a weighted combination of F1 and Accuracy
            method_data['combined_score'] = (method_data['macro_f1'] + method_data['accuracy']) / 2
            best_idx = method_data['combined_score'].idxmax()
            best_row = method_data.loc[best_idx]
            final_results[method] = {
                'final_f1': best_row['macro_f1'],
                'final_accuracy': best_row['accuracy'],
                'final_iteration': best_row['iteration_no'],
                'final_labels': best_row.get('train_data_size', best_row['iteration_no'] * 1000),
                'combined_score': best_row['combined_score']
            }
    
    # Find best method (excluding random) - using combined score
    non_random_methods = {k: v for k, v in final_results.items() if 'random' not in k.lower()}
    
    if not non_random_methods:
        return {"error": "No non-random methods found"}
    
    best_method = max(non_random_methods.keys(), key=lambda x: non_random_methods[x]['combined_score'])
    best_f1 = non_random_methods[best_method]['final_f1']
    best_accuracy = non_random_methods[best_method]['final_accuracy']
    best_combined = non_random_methods[best_method]['combined_score']
    
    # Find random sampling method
    random_method = None
    for method in final_results.keys():
        if 'random' in method.lower():
            random_method = method
            break
    
    if not random_method:
        return {"error": "No random sampling method found"}
    
    random_f1 = final_results[random_method]['final_f1']
    random_accuracy = final_results[random_method]['final_accuracy']
    random_combined = final_results[random_method]['combined_score']
    
    # Analysis 1: Performance difference at same data amount (using combined score)
    performance_diff_pct = ((best_combined - random_combined) / random_combined) * 100 if random_combined > 0 else 0
    
    # Analysis 2: Data efficiency - how much data does best method need to reach random's final performance
    target_combined = random_combined * 0.95  # 95% of random's final combined performance
    
    # Get all data for best method
    best_method_data = valid_df[valid_df['method'] == best_method].copy()
    best_method_data['combined_score'] = (best_method_data['macro_f1'] + best_method_data['accuracy']) / 2
    best_method_data = best_method_data.sort_values('iteration_no')
    
    data_for_target = None
    for _, row in best_method_data.iterrows():
        if row['combined_score'] >= target_combined:
            data_for_target = row.get('train_data_size', row['iteration_no'] * 1000)
            break
    
    if data_for_target is None:
        data_for_target = non_random_methods[best_method]['final_labels']
    
    random_final_labels = final_results[random_method]['final_labels']
    data_savings_pct = ((random_final_labels - data_for_target) / random_final_labels) * 100 if random_final_labels > 0 else 0
    
    # Analysis 3: How much data does random need to reach best method's final performance
    target_combined_best = best_combined * 0.95  # 95% of best method's final combined performance
    
    # Get all data for random method
    random_method_data = valid_df[valid_df['method'] == random_method].copy()
    random_method_data['combined_score'] = (random_method_data['macro_f1'] + random_method_data['accuracy']) / 2
    random_method_data = random_method_data.sort_values('iteration_no')
    
    data_for_random_target = None
    for _, row in random_method_data.iterrows():
        if row['combined_score'] >= target_combined_best:
            data_for_random_target = row.get('train_data_size', row['iteration_no'] * 1000)
            break
    
    if data_for_random_target is None:
        data_for_random_target = random_final_labels
    
    best_final_labels = non_random_methods[best_method]['final_labels']
    data_ratio = data_for_random_target / best_final_labels if best_final_labels > 0 else 1
    
    return {
        "best_method": best_method,
        "random_method": random_method,
        "best_final_f1": best_f1,
        "best_final_accuracy": best_accuracy,
        "best_combined_score": best_combined,
        "random_final_f1": random_f1,
        "random_final_accuracy": random_accuracy,
        "random_combined_score": random_combined,
        "performance_difference_pct": performance_diff_pct,
        "data_for_target": data_for_target,
        "random_final_labels": random_final_labels,
        "data_savings_pct": data_savings_pct,
        "data_for_random_target": data_for_random_target,
        "best_final_labels": best_final_labels,
        "data_ratio": data_ratio,
        "all_final_results": final_results
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Active Learning Analysis for Project Objectives")
    parser.add_argument("--input", default="results/results_final/results_N3000.csv", help="Input results file")
    parser.add_argument("--output", default="analysis_results.txt", help="Output file for results")
    
    args = parser.parse_args()
    
    # Read and prepare data
    df = _read_results(args.input)
    
    if "metrics" not in df.columns:
        print("Error: Input file must contain a 'metrics' column")
        return 1
    
    # Analyze all tests combined
    print("=== COMBINED ANALYSIS ===")
    result = analyze_all_tests_combined(df)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    
    print(f"Best Method: {result['best_method']}")
    print(f"Random Method: {result['random_method']}")
    print(f"Best Final F1: {result['best_final_f1']:.4f}")
    print(f"Best Final Accuracy: {result['best_final_accuracy']:.4f}")
    print(f"Best Combined Score: {result['best_combined_score']:.4f}")
    print(f"Random Final F1: {result['random_final_f1']:.4f}")
    print(f"Random Final Accuracy: {result['random_final_accuracy']:.4f}")
    print(f"Random Combined Score: {result['random_combined_score']:.4f}")
    print(f"Performance Difference: {result['performance_difference_pct']:.2f}%")
    print(f"Data Savings: {result['data_savings_pct']:.2f}%")
    print(f"Data Ratio (Random/Best): {result['data_ratio']:.2f}x")
    
    print(f"\n=== ALL FINAL RESULTS ===")
    for method, data in result['all_final_results'].items():
        print(f"{method}: F1={data['final_f1']:.4f}, Acc={data['final_accuracy']:.4f}, Combined={data['combined_score']:.4f}, Labels={data['final_labels']}")
    
    # Save results
    with open(args.output, 'w') as f:
        f.write("=== AKTIF OĞRENME ANALIZ SONUCLARI ===\n\n")
        
        f.write("PROJE HEDEFLERI CEVAPLARI:\n\n")
        
        f.write(f"1. RASTGELE ÖRNEKLEMEYE KIYASLA BAŞARI FARKI: %{result['performance_difference_pct']:.2f}\n")
        f.write(f"   - En Iyi Yöntem: {result['best_method']}\n")
        f.write(f"   - En Iyi F1: {result['best_final_f1']:.4f}\n")
        f.write(f"   - Rastgele F1: {result['random_final_f1']:.4f}\n\n")
        
        f.write(f"2. AYNI BAŞARI IÇIN GEREKLI VERI MIKTARI: %{result['data_savings_pct']:.2f} TASARRUF\n")
        f.write(f"   - En Iyi Yöntem {result['random_final_labels'] - result['data_for_target']:.0f} daha az etiket gerektiriyor\n")
        f.write(f"   - En Iyi: {result['data_for_target']:.0f} etiket\n")
        f.write(f"   - Rastgele: {result['random_final_labels']:.0f} etiket\n\n")
        
        f.write(f"3. TAM ETIKETLI VERI ILE KARŞILAŞTIRMA:\n")
        f.write(f"   - Rastgele örnekleme, en iyi yöntemin performansına ulaşmak için {result['data_ratio']:.2f}x daha fazla veri gerektiriyor\n")
        f.write(f"   - Rastgele gereken: {result['data_for_random_target']:.0f} etiket\n")
        f.write(f"   - En iyi gereken: {result['best_final_labels']:.0f} etiket\n\n")
        
        f.write("=== TÜM YÖNTEMLERIN SONUCLARI ===\n")
        for method, data in result['all_final_results'].items():
            f.write(f"{method}: F1={data['final_f1']:.4f}, Labels={data['final_labels']}\n")
    
    print(f"\nResults saved to: {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
