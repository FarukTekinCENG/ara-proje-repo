#!/usr/bin/env python3
"""
İstatistiksel Analiz Scripti
Mevcut plot_graph.py çıktılarını kullanarak istatistiksel testler yapar
"""

import pandas as pd
import numpy as np
from scipy import stats
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

def statistical_analysis(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Perform statistical analysis on active learning results"""
    
    methods = df['method'].unique()
    results = {}
    
    # 1. Paired t-test between active learning and random
    if 'random_sampling' in methods or 'random' in methods:
        random_method = 'random_sampling' if 'random_sampling' in methods else 'random'
        random_data = df[df['method'] == random_method]
        
        for method in methods:
            if method not in [random_method, 'base_classifier', 'BASE']:
                method_data = df[df['method'] == method]
                
                # Align iterations
                min_iter = min(len(random_data), len(method_data))
                random_f1 = random_data['macro_f1'].iloc[:min_iter]
                method_f1 = method_data['macro_f1'].iloc[:min_iter]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(method_f1, random_f1)
                
                # Effect size (Cohen's d)
                diff = method_f1 - random_f1
                pooled_std = np.sqrt(((len(method_f1)-1)*method_f1.var() + 
                                    (len(random_f1)-1)*random_f1.var()) / 
                                   (len(method_f1) + len(random_f1) - 2))
                cohens_d = diff.mean() / pooled_std if pooled_std > 0 else 0
                
                results[f"{method}_vs_{random_method}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'mean_diff': diff.mean(),
                    'mean_diff_pct': (diff.mean() / random_f1.mean()) * 100 if random_f1.mean() > 0 else 0,
                    'method_final_f1': method_f1.iloc[-1],
                    'random_final_f1': random_f1.iloc[-1]
                }
    
    # 2. ANOVA test for multiple methods
    method_groups = []
    valid_methods = [m for m in methods if m not in ['base_classifier', 'BASE']]
    
    for method in valid_methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 1:
            method_groups.append(method_data['macro_f1'].values)
    
    if len(method_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*method_groups)
        results['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'n_methods': len(method_groups)
        }
    
    # 3. Data efficiency calculation
    efficiency_results = {}
    if len(df) > 0:
        max_f1 = df['macro_f1'].max()
        target_performance = max_f1 * 0.95  # 95% of max performance
        
        for method in valid_methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                # Find iteration reaching target performance
                target_iter = None
                for _, row in method_data.iterrows():
                    if row['macro_f1'] >= target_performance:
                        target_iter = row['iteration_no']
                        break
                
                # Random baseline
                if random_method in methods:
                    random_data = df[df['method'] == random_method]
                    random_iter = None
                    for _, row in random_data.iterrows():
                        if row['macro_f1'] >= target_performance:
                            random_iter = row['iteration_no']
                            break
                    
                    if target_iter and random_iter:
                        data_reduction = ((random_iter - target_iter) / random_iter) * 100
                        efficiency_results[method] = {
                            'target_iter': target_iter,
                            'random_iter': random_iter,
                            'data_reduction_pct': data_reduction,
                            'efficiency_ratio': random_iter / target_iter if target_iter > 0 else 0
                        }
    
    return results, efficiency_results

def plot_statistical_results(results: Dict[str, Any], efficiency_results: Dict[str, Any], 
                           output_dir: str = "plots"):
    """Create statistical analysis visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance improvement percentages
    ax1 = axes[0, 0]
    methods = []
    improvements = []
    p_values = []
    
    for key, result in results.items():
        if key.endswith('_vs_random_sampling') or key.endswith('_vs_random'):
            method = key.replace('_vs_random_sampling', '').replace('_vs_random', '')
            methods.append(method)
            improvements.append(result['mean_diff_pct'])
            p_values.append(result['p_value'])
    
    if methods:
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        bars = ax1.bar(methods, improvements, color=colors, alpha=0.7)
        
        # Add significance stars
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            if p_val < 0.001:
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(improvements)*0.02, '***', ha='center', va='bottom')
            elif p_val < 0.01:
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(improvements)*0.02, '**', ha='center', va='bottom')
            elif p_val < 0.05:
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(improvements)*0.02, '*', ha='center', va='bottom')
        
        ax1.set_ylabel('Performans İyileşmesi (\%)')
        ax1.set_title('Rastgele Örnekleme Karşısında Performans Farkı')
        ax1.grid(True, alpha=0.3)
    
    # 2. Data efficiency comparison
    ax2 = axes[0, 1]
    if efficiency_results:
        methods_eff = list(efficiency_results.keys())
        reductions = [efficiency_results[method]['data_reduction_pct'] for method in methods_eff]
        
        bars = ax2.bar(methods_eff, reductions, color='skyblue', alpha=0.8)
        ax2.set_ylabel('Veri Tasarrufu (\%)')
        ax2.set_title('Aynı Performans için Veri Verimliliği')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, reduction in zip(bars, reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(reductions)*0.02, 
                    f'{reduction:.1f}%', ha='center', va='bottom')
    
    # 3. Effect sizes
    ax3 = axes[1, 0]
    effect_sizes = []
    effect_methods = []
    
    for key, result in results.items():
        if key.endswith('_vs_random_sampling') or key.endswith('_vs_random'):
            method = key.replace('_vs_random_sampling', '').replace('_vs_random', '')
            effect_methods.append(method)
            effect_sizes.append(result['cohens_d'])
    
    if effect_methods:
        colors = ['red' if abs(d) < 0.2 else 'orange' if abs(d) < 0.5 else 'green' for d in effect_sizes]
        bars = ax3.bar(effect_methods, effect_sizes, color=colors, alpha=0.7)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Küçük etki')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Orta etki')
        ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Büyük etki')
        
        ax3.set_ylabel("Cohen's d")
        ax3.set_title('Etki Büyüklüğü (Cohen\'s d)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary data
    summary_data = []
    for method in efficiency_results.keys():
        key = f"{method}_vs_random_sampling" if f"{method}_vs_random_sampling" in results else f"{method}_vs_random"
        if key in results:
            summary_data.append([
                method,
                f"{results[key]['mean_diff_pct']:.2f}\%",
                f"{efficiency_results[method]['data_reduction_pct']:.1f}\%",
                f"{results[key]['p_value']:.4f}",
                f"{results[key]['cohens_d']:.3f}"
            ])
    
    if summary_data:
        table = ax4.table(cellText=summary_data,
                         colLabels=['Strateji', 'Performans Farkı', 'Veri Tasarrufu', 'p-değeri', 'Etki Büyüklüğü'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code p-values
        for i in range(len(summary_data)):
            p_val = float(summary_data[i][3])
            if p_val < 0.05:
                for j in range(5):
                    table[(i+1, j)].set_facecolor('#e8f5e8')
            elif p_val < 0.1:
                for j in range(5):
                    table[(i+1, j)].set_facecolor('#fff3cd')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/statistical_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_tables(results: Dict[str, Any], efficiency_results: Dict[str, Any], 
                          output_dir: str = "plots"):
    """Generate summary tables in CSV format"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Performance comparison table
    performance_data = []
    for key, result in results.items():
        if key.endswith('_vs_random_sampling') or key.endswith('_vs_random'):
            method = key.replace('_vs_random_sampling', '').replace('_vs_random', '')
            performance_data.append({
                'Strateji': method,
                'Ortalama Fark': f"{result['mean_diff']:.4f}",
                'Yüzde Fark': f"{result['mean_diff_pct']:.2f}%",
                'p-değeri': f"{result['p_value']:.4f}",
                'Etki Büyüklüğü': f"{result['cohens_d']:.3f}",
                'Anlamlılık': '***' if result['p_value'] < 0.001 else 
                           '**' if result['p_value'] < 0.01 else
                           '*' if result['p_value'] < 0.05 else 'ns'
            })
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(f"{output_dir}/performance_comparison.csv", index=False)
    
    # Efficiency table
    efficiency_data = []
    for method, result in efficiency_results.items():
        efficiency_data.append({
            'Strateji': method,
            'Hedef İterasyon': result['target_iter'],
            'Rastgele İterasyon': result['random_iter'],
            'Veri Tasarrufu (%)': f"{result['data_reduction_pct']:.1f}%",
            'Verimlilik Oranı': f"{result['efficiency_ratio']:.2f}x"
        })
    
    if efficiency_data:
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df.to_csv(f"{output_dir}/efficiency_comparison.csv", index=False)
    
    return performance_data, efficiency_data

def main() -> int:
    parser = argparse.ArgumentParser(description="Statistical Analysis for Active Learning Results")
    parser.add_argument("--input", default="results/results.xlsx", help="Input results file")
    parser.add_argument("--output_dir", default="statistical_analysis", help="Output directory")
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
        
        # Perform statistical analysis
        results, efficiency_results = statistical_analysis(valid_df)
        
        # Generate plots
        plot_statistical_results(results, efficiency_results, output_dir)
        
        # Generate tables
        generate_summary_tables(results, efficiency_results, output_dir)
        
        print(f"Statistical analysis completed for test {idx}")
        print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
