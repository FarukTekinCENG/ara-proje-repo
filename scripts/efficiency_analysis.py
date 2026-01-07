#!/usr/bin/env python3
"""
Verimlilik Analiz Scripti
Aktif öğrenme stratejilerinin verimliliğini ve maliyet-etkinliğini analiz eder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Split dataframe into test blocks"""
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

def calculate_efficiency_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate efficiency metrics for each method"""
    
    methods = df['method'].unique()
    efficiency_results = {}
    
    # Get overall best performance
    all_f1_scores = []
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            all_f1_scores.extend(method_data['macro_f1'].values)
    
    if not all_f1_scores:
        return efficiency_results
    
    max_f1 = max(all_f1_scores)
    target_performance_levels = [0.7, 0.8, 0.9, 0.95]  # Different performance targets
    
    for method in methods:
        if method in ['base_classifier', 'BASE']:
            continue
            
        method_data = df[df['method'] == method].copy()
        if len(method_data) == 0:
            continue
        
        method_results = {}
        
        # 1. Learning efficiency (Area Under Curve)
        if len(method_data) > 1:
            auc = np.trapezoid(method_data['macro_f1'], method_data['iteration_no'])
            total_labels = method_data['train_data_size'].iloc[-1] if 'train_data_size' in method_data.columns else method_data['iteration_no'].iloc[-1] * 1000
            learning_efficiency = auc / total_labels * 1000 if total_labels > 0 else 0
            method_results['learning_efficiency'] = learning_efficiency
            method_results['auc'] = auc
            method_results['total_labels'] = total_labels
        
        # 2. Convergence analysis for different performance targets
        for target_pct in target_performance_levels:
            target_f1 = max_f1 * target_pct
            
            # Find iteration reaching target
            target_iter = None
            target_labels = None
            
            for _, row in method_data.iterrows():
                if row['macro_f1'] >= target_f1:
                    target_iter = row['iteration_no']
                    target_labels = row.get('train_data_size', row['iteration_no'] * 1000)
                    break
            
            if target_iter is not None:
                method_results[f'converge_{int(target_pct*100)}'] = {
                    'iteration': target_iter,
                    'labels': target_labels,
                    'performance': row['macro_f1']
                }
        
        # 3. Label efficiency (performance per 1000 labels)
        if 'train_data_size' in method_data.columns:
            final_performance = method_data['macro_f1'].iloc[-1]
            final_labels = method_data['train_data_size'].iloc[-1]
            label_efficiency = (final_performance / final_labels) * 1000 if final_labels > 0 else 0
            method_results['label_efficiency'] = label_efficiency
        
        # 4. Early performance (first 25% of iterations)
        if len(method_data) >= 4:
            early_iter = len(method_data) // 4
            early_performance = method_data['macro_f1'].iloc[early_iter]
            method_results['early_performance'] = early_performance
            method_results['early_iteration'] = early_iter
        
        efficiency_results[method] = method_results
    
    return efficiency_results

def calculate_cost_benefit_analysis(df: pd.DataFrame, efficiency_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate cost-benefit analysis"""
    
    cost_benefit_results = {}
    
    # Find random sampling baseline
    random_method = None
    for method in df['method'].unique():
        if 'random' in method.lower():
            random_method = method
            break
    
    if not random_method:
        return cost_benefit_results
    
    random_data = df[df['method'] == random_method]
    if len(random_data) == 0:
        return cost_benefit_results
    
    # Get final random performance
    random_final_f1 = random_data['macro_f1'].iloc[-1]
    random_final_labels = random_data.get('train_data_size', pd.Series([len(random_data) * 1000])).iloc[-1]
    
    for method, results in efficiency_results.items():
        if method == random_method:
            continue
        
        method_data = df[df['method'] == method]
        if len(method_data) == 0:
            continue
        
        method_final_f1 = method_data['macro_f1'].iloc[-1]
        method_final_labels = method_data.get('train_data_size', pd.Series([len(method_data) * 1000])).iloc[-1]
        
        # Cost-benefit metrics
        performance_gain = method_final_f1 - random_final_f1
        performance_gain_pct = (performance_gain / random_final_f1) * 100 if random_final_f1 > 0 else 0
        
        label_savings = random_final_labels - method_final_labels
        label_savings_pct = (label_savings / random_final_labels) * 100 if random_final_labels > 0 else 0
        
        # Efficiency ratio
        efficiency_ratio = (method_final_f1 / method_final_labels) / (random_final_f1 / random_final_labels) if random_final_f1 > 0 and method_final_labels > 0 else 1
        
        cost_benefit_results[method] = {
            'performance_gain': performance_gain,
            'performance_gain_pct': performance_gain_pct,
            'label_savings': label_savings,
            'label_savings_pct': label_savings_pct,
            'efficiency_ratio': efficiency_ratio,
            'random_f1': random_final_f1,
            'method_f1': method_final_f1,
            'random_labels': random_final_labels,
            'method_labels': method_final_labels
        }
    
    return cost_benefit_results

def plot_efficiency_analysis(efficiency_results: Dict[str, Any], 
                           cost_benefit_results: Dict[str, Any],
                           output_dir: str = "plots"):
    """Create efficiency analysis visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = list(efficiency_results.keys())
    
    # 1. Learning Efficiency
    ax1 = axes[0, 0]
    learning_efficiencies = []
    for method in methods:
        if 'learning_efficiency' in efficiency_results[method]:
            learning_efficiencies.append(efficiency_results[method]['learning_efficiency'])
        else:
            learning_efficiencies.append(0)
    
    bars = ax1.bar(methods, learning_efficiencies, color='lightblue', alpha=0.8)
    ax1.set_ylabel('Öğrenme Verimliliği (AUC/Etiket)')
    ax1.set_title('Öğrenme Verimliliği Karşılaştırması')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, efficiency in zip(bars, learning_efficiencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(learning_efficiencies)*0.02, 
                f'{efficiency:.3f}', ha='center', va='bottom')
    
    # 2. Convergence Speed (95% of max performance)
    ax2 = axes[0, 1]
    convergence_iters = []
    for method in methods:
        if 'converge_95' in efficiency_results[method]:
            convergence_iters.append(efficiency_results[method]['converge_95']['iteration'])
        else:
            convergence_iters.append(np.nan)
    
    valid_methods = [m for m, i in zip(methods, convergence_iters) if not np.isnan(i)]
    valid_iters = [i for i in convergence_iters if not np.isnan(i)]
    
    if valid_methods:
        bars = ax2.bar(valid_methods, valid_iters, color='lightgreen', alpha=0.8)
        ax2.set_ylabel('Yakınsama İterasyonu')
        ax2.set_title('95% Performans Yakınsama Hızı')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, iter_count in zip(bars, valid_iters):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(valid_iters)*0.02, 
                    f'{int(iter_count)}', ha='center', va='bottom')
    
    # 3. Label Efficiency
    ax3 = axes[0, 2]
    label_efficiencies = []
    for method in methods:
        if 'label_efficiency' in efficiency_results[method]:
            label_efficiencies.append(efficiency_results[method]['label_efficiency'])
        else:
            label_efficiencies.append(0)
    
    bars = ax3.bar(methods, label_efficiencies, color='lightcoral', alpha=0.8)
    ax3.set_ylabel('Etiket Verimliliği (Performans/1000 Etiket)')
    ax3.set_title('Etiket Verimliliği Karşılaştırması')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, efficiency in zip(bars, label_efficiencies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(label_efficiencies)*0.02, 
                f'{efficiency:.4f}', ha='center', va='bottom')
    
    # 4. Performance Gain Percentage
    ax4 = axes[1, 0]
    if cost_benefit_results:
        gain_methods = list(cost_benefit_results.keys())
        gains = [cost_benefit_results[method]['performance_gain_pct'] for method in gain_methods]
        
        colors = ['green' if g > 5 else 'orange' if g > 0 else 'red' for g in gains]
        bars = ax4.bar(gain_methods, gains, color=colors, alpha=0.8)
        ax4.set_ylabel('Performans Kazancı (\%)')
        ax4.set_title('Rastgele Örnekleme Karşısında Performans Kazancı')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, gain in zip(bars, gains):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(gains)*0.02, 
                    f'{gain:.2f}%', ha='center', va='bottom')
    
    # 5. Label Savings Percentage
    ax5 = axes[1, 1]
    if cost_benefit_results:
        savings_methods = list(cost_benefit_results.keys())
        savings = [cost_benefit_results[method]['label_savings_pct'] for method in savings_methods]
        
        colors = ['green' if s > 10 else 'orange' if s > 0 else 'red' for s in savings]
        bars = ax5.bar(savings_methods, savings, color=colors, alpha=0.8)
        ax5.set_ylabel('Etiket Tasarrufu (\%)')
        ax5.set_title('Aynı Performans için Etiket Tasarrufu')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, saving in zip(bars, savings):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(savings)*0.02, 
                    f'{saving:.1f}%', ha='center', va='bottom')
    
    # 6. Efficiency Ratio
    ax6 = axes[1, 2]
    if cost_benefit_results:
        ratio_methods = list(cost_benefit_results.keys())
        ratios = [cost_benefit_results[method]['efficiency_ratio'] for method in ratio_methods]
        
        colors = ['green' if r > 1.2 else 'orange' if r > 1.0 else 'red' for r in ratios]
        bars = ax6.bar(ratio_methods, ratios, color=colors, alpha=0.8)
        ax6.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Rastgele Seviyesi')
        ax6.set_ylabel('Verimlilik Oranı')
        ax6.set_title('Rastgele Örnekleme Göre Verimlilik Oranı')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max(ratios)*0.02, 
                    f'{ratio:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_efficiency_tables(efficiency_results: Dict[str, Any], 
                             cost_benefit_results: Dict[str, Any],
                             output_dir: str = "plots"):
    """Generate efficiency analysis tables"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Efficiency summary table
    efficiency_data = []
    for method, results in efficiency_results.items():
        row = {'Strateji': method}
        
        if 'learning_efficiency' in results:
            row['Öğrenme Verimliliği'] = f"{results['learning_efficiency']:.4f}"
        
        if 'converge_95' in results:
            row['95% Yakınsama'] = results['converge_95']['iteration']
        
        if 'label_efficiency' in results:
            row['Etiket Verimliliği'] = f"{results['label_efficiency']:.4f}"
        
        if 'early_performance' in results:
            row['Erken Performans'] = f"{results['early_performance']:.4f}"
        
        efficiency_data.append(row)
    
    if efficiency_data:
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df.to_csv(f"{output_dir}/efficiency_summary.csv", index=False)
    
    # Cost-benefit summary table
    cost_benefit_data = []
    if cost_benefit_results:
        for method, results in cost_benefit_results.items():
            cost_benefit_data.append({
                'Strateji': method,
                'Performans Kazancı (%)': f"{results['performance_gain_pct']:.2f}",
                'Etiket Tasarrufu (%)': f"{results['label_savings_pct']:.1f}",
                'Verimlilik Oranı': f"{results['efficiency_ratio']:.2f}x",
                'Final F1': f"{results['method_f1']:.4f}",
                'Final Etiket': int(results['method_labels'])
            })
        
        cost_benefit_df = pd.DataFrame(cost_benefit_data)
        cost_benefit_df.to_csv(f"{output_dir}/cost_benefit_summary.csv", index=False)
    
    return efficiency_data, cost_benefit_data

def main() -> int:
    parser = argparse.ArgumentParser(description="Efficiency Analysis for Active Learning Results")
    parser.add_argument("--input", default="results/results.xlsx", help="Input results file")
    parser.add_argument("--output_dir", default="efficiency_analysis", help="Output directory")
    parser.add_argument("--test_index", type=int, default=None, help="Test block index (1-based)")
    parser.add_argument("--no_split", action="store_true", help="Treat as single test block")
    
    args = parser.parse_args()
    
    # Read and prepare data
    df = _read_results(args.input)
    
    if "metrics" not in df.columns:
        raise KeyError("Input file must contain a 'metrics' column")
    
    df["metrics_dict"] = df["metrics"].apply(_safe_json_loads)
    
    # Extract macro_f1 from metrics
    macro_f1_values = []
    for metrics_dict in df["metrics_dict"]:
        if isinstance(metrics_dict, dict) and "macro_f1" in metrics_dict:
            macro_f1_values.append(metrics_dict["macro_f1"])
        else:
            macro_f1_values.append(np.nan)
    
    df["macro_f1"] = macro_f1_values
    
    # Split into tests if needed
    tests = [df.reset_index(drop=True)] if args.no_split else _split_into_tests(df)
    
    if args.test_index is not None:
        if args.test_index <= 0 or args.test_index > len(tests):
            raise ValueError(f"--test_index must be between 1 and {len(tests)} (got {args.test_index})")
        tests = [tests[args.test_index - 1]]
    
    # Process each test
    for idx, test_df in enumerate(tests, 1):
        output_dir = os.path.join(args.output_dir, f"test_{idx}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out base classifier rows
        valid_df = test_df[~test_df['method'].isin(['base_classifier', 'BASE'])].copy()
        
        if len(valid_df) == 0:
            print(f"Warning: No valid data found for test {idx}")
            continue
        
        # Calculate efficiency metrics
        efficiency_results = calculate_efficiency_metrics(valid_df)
        
        # Calculate cost-benefit analysis
        cost_benefit_results = calculate_cost_benefit_analysis(valid_df, efficiency_results)
        
        # Generate plots
        plot_efficiency_analysis(efficiency_results, cost_benefit_results, output_dir)
        
        # Generate tables
        generate_efficiency_tables(efficiency_results, cost_benefit_results, output_dir)
        
        print(f"Efficiency analysis completed for test {idx}")
        print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())