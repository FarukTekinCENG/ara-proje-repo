#!/usr/bin/env python3
"""
Comprehensive Active Learning Analysis Script
Calculates performance improvements and data efficiency metrics
Generates PDF report with statistical analysis
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ActiveLearningAnalyzer:
    def __init__(self):
        self.full_labeled_data = {
            'accuracy': 0.716,
            'macro_f1': 0.725,
            'train_data_size': 1000,
            'method': 'random_sampling'
        }
        
    def load_and_parse_data(self, file_path: str) -> pd.DataFrame:
        """Load and parse Excel/CSV results file"""
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Parse metrics column
        def parse_metrics(x):
            try:
                if pd.isna(x):
                    return {}
                return json.loads(str(x))
            except:
                return {}
        
        df['metrics_dict'] = df['metrics'].apply(parse_metrics)
        return df
    
    def extract_performance_curves(self, df: pd.DataFrame) -> Dict[str, Dict[str, List]]:
        """Extract performance curves for each method"""
        curves = {}
        
        for method in df['method'].unique():
            if pd.isna(method) or method == 'BASE':
                continue
                
            method_df = df[df['method'] == method].copy()
            
            # Sort by train data size
            method_df = method_df.sort_values('train_data_size')
            
            # Extract metrics
            train_sizes = method_df['train_data_size'].tolist()
            accuracies = [m.get('accuracy', 0) for m in method_df['metrics_dict']]
            macro_f1s = [m.get('macro_f1', 0) for m in method_df['metrics_dict']]
            
            curves[method] = {
                'train_sizes': train_sizes,
                'accuracies': accuracies,
                'macro_f1s': macro_f1s
            }
        
        return curves
    
    def calculate_performance_improvement(self, curves: Dict) -> Dict[str, Dict]:
        """Calculate performance improvement over random sampling"""
        improvements = {}
        
        if 'random_sampling' not in curves:
            return improvements
        
        random_curve = curves['random_sampling']
        
        for method, curve in curves.items():
            if method == 'random_sampling':
                continue
            
            # Find common train sizes
            random_sizes = set(random_curve['train_sizes'])
            method_sizes = set(curve['train_sizes'])
            common_sizes = sorted(list(random_sizes & method_sizes))
            
            if not common_sizes:
                continue
            
            # Calculate improvements at common sizes
            acc_improvements = []
            f1_improvements = []
            
            for size in common_sizes:
                random_idx = random_curve['train_sizes'].index(size)
                method_idx = curve['train_sizes'].index(size)
                
                random_acc = random_curve['accuracies'][random_idx]
                method_acc = curve['accuracies'][method_idx]
                acc_improvement = ((method_acc - random_acc) / random_acc) * 100
                acc_improvements.append(acc_improvement)
                
                random_f1 = random_curve['macro_f1s'][random_idx]
                method_f1 = curve['macro_f1s'][method_idx]
                f1_improvement = ((method_f1 - random_f1) / random_f1) * 100
                f1_improvements.append(f1_improvement)
            
            improvements[method] = {
                'accuracy_improvement_pct': np.mean(acc_improvements),
                'macro_f1_improvement_pct': np.mean(f1_improvements),
                'max_accuracy_improvement_pct': max(acc_improvements),
                'max_macro_f1_improvement_pct': max(f1_improvements),
                'common_sizes': common_sizes
            }
        
        return improvements
    
    def calculate_data_efficiency(self, curves: Dict) -> Dict[str, Dict]:
        """Calculate data efficiency - how much less data needed to achieve same performance"""
        efficiency = {}
        
        if 'random_sampling' not in curves:
            return efficiency
        
        random_curve = curves['random_sampling']
        
        for method, curve in curves.items():
            if method == 'random_sampling':
                continue
            
            # Target performance levels
            target_accuracies = [0.3, 0.4, 0.5, 0.6]
            target_f1s = [0.3, 0.4, 0.5, 0.6]
            
            acc_savings = []
            f1_savings = []
            
            for target in target_accuracies:
                random_data_needed = self._find_data_for_target(random_curve, target, 'accuracy')
                method_data_needed = self._find_data_for_target(curve, target, 'accuracy')
                
                if random_data_needed and method_data_needed:
                    savings = ((random_data_needed - method_data_needed) / random_data_needed) * 100
                    acc_savings.append(savings)
            
            for target in target_f1s:
                random_data_needed = self._find_data_for_target(random_curve, target, 'macro_f1')
                method_data_needed = self._find_data_for_target(curve, target, 'macro_f1')
                
                if random_data_needed and method_data_needed:
                    savings = ((random_data_needed - method_data_needed) / random_data_needed) * 100
                    f1_savings.append(savings)
            
            efficiency[method] = {
                'avg_accuracy_data_savings_pct': np.mean(acc_savings) if acc_savings else 0,
                'avg_macro_f1_data_savings_pct': np.mean(f1_savings) if f1_savings else 0,
                'max_accuracy_data_savings_pct': max(acc_savings) if acc_savings else 0,
                'max_macro_f1_data_savings_pct': max(f1_savings) if f1_savings else 0
            }
        
        return efficiency
    
    def _find_data_for_target(self, curve: Dict, target: float, metric: str) -> int:
        """Find train data size needed to reach target performance"""
        if metric == 'accuracy':
            values = curve['accuracies']
        elif metric == 'macro_f1':
            values = curve['macro_f1s']
        else:
            return None
        
        sizes = curve['train_sizes']
        
        for size, value in zip(sizes, values):
            if value >= target:
                return size
        
        return None
    
    def compare_with_full_labeled(self, curves: Dict) -> Dict[str, Dict]:
        """Compare with full labeled data performance"""
        comparisons = {}
        
        full_acc = self.full_labeled_data['accuracy']
        full_f1 = self.full_labeled_data['macro_f1']
        
        for method, curve in curves.items():
            final_acc = curve['accuracies'][-1]
            final_f1 = curve['macro_f1s'][-1]
            final_size = curve['train_sizes'][-1]
            
            acc_gap = ((full_acc - final_acc) / full_acc) * 100
            f1_gap = ((full_f1 - final_f1) / full_f1) * 100
            
            comparisons[method] = {
                'final_accuracy': final_acc,
                'final_macro_f1': final_f1,
                'final_train_size': final_size,
                'accuracy_gap_to_full_pct': acc_gap,
                'macro_f1_gap_to_full_pct': f1_gap,
                'data_efficiency_vs_full': (final_size / self.full_labeled_data['train_data_size']) * 100
            }
        
        return comparisons
    
    def statistical_analysis(self, curves: Dict) -> Dict[str, Any]:
        """Perform statistical analysis"""
        if 'random_sampling' not in curves or len(curves) < 2:
            return {}
        
        # Collect final performances
        methods = list(curves.keys())
        final_accuracies = [curves[method]['accuracies'][-1] for method in methods]
        final_f1s = [curves[method]['macro_f1s'][-1] for method in methods]
        
        # Basic statistics
        stats_results = {
            'methods': methods,
            'accuracy_stats': {
                'mean': np.mean(final_accuracies),
                'std': np.std(final_accuracies),
                'min': np.min(final_accuracies),
                'max': np.max(final_accuracies)
            },
            'macro_f1_stats': {
                'mean': np.mean(final_f1s),
                'std': np.std(final_f1s),
                'min': np.min(final_f1s),
                'max': np.max(final_f1s)
            }
        }
        
        # Pairwise comparisons with random sampling
        if 'random_sampling' in methods:
            random_idx = methods.index('random_sampling')
            random_acc = final_accuracies[random_idx]
            random_f1 = final_f1s[random_idx]
            
            for i, method in enumerate(methods):
                if method == 'random_sampling':
                    continue
                
                # Simple t-test (note: small sample size, so this is approximate)
                other_acc = final_accuracies[i]
                other_f1 = final_f1s[i]
                
                acc_diff = other_acc - random_acc
                f1_diff = other_f1 - random_f1
                
                stats_results[f'{method}_vs_random'] = {
                    'accuracy_difference': acc_diff,
                    'macro_f1_difference': f1_diff,
                    'accuracy_improvement_pct': (acc_diff / random_acc) * 100,
                    'macro_f1_improvement_pct': (f1_diff / random_f1) * 100
                }
        
        return stats_results
    
    def create_plots(self, curves: Dict, improvements: Dict, efficiency: Dict, 
                    comparisons: Dict, stats: Dict) -> List[plt.Figure]:
        """Create all plots for the report"""
        figures = []
        
        # Plot 1: Performance Curves
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for method, curve in curves.items():
            ax1.plot(curve['train_sizes'], curve['accuracies'], 
                    marker='o', label=method.replace('_', ' ').title(), linewidth=2)
            ax2.plot(curve['train_sizes'], curve['macro_f1s'], 
                    marker='s', label=method.replace('_', ' ').title(), linewidth=2)
        
        # Add full labeled data reference
        ax1.axhline(y=self.full_labeled_data['accuracy'], 
                   color='red', linestyle='--', label='Full Labeled Data')
        ax2.axhline(y=self.full_labeled_data['macro_f1'], 
                   color='red', linestyle='--', label='Full Labeled Data')
        
        ax1.set_xlabel('Training Data Size')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Training Data Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Training Data Size')
        ax2.set_ylabel('Macro F1')
        ax2.set_title('Macro F1 vs Training Data Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig1.suptitle('Active Learning Performance Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        figures.append(fig1)
        
        # Plot 2: Performance Improvements
        if improvements:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            methods = list(improvements.keys())
            acc_improvements = [improvements[m]['accuracy_improvement_pct'] for m in methods]
            f1_improvements = [improvements[m]['macro_f1_improvement_pct'] for m in methods]
            
            bars1 = ax1.bar(methods, acc_improvements, color='skyblue', alpha=0.7)
            ax1.set_ylabel('Accuracy Improvement (%)')
            ax1.set_title('Accuracy Improvement Over Random Sampling')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, acc_improvements):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            bars2 = ax2.bar(methods, f1_improvements, color='lightcoral', alpha=0.7)
            ax2.set_ylabel('Macro F1 Improvement (%)')
            ax2.set_title('Macro F1 Improvement Over Random Sampling')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars2, f1_improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            fig2.suptitle('Performance Improvements vs Random Sampling', fontsize=16, fontweight='bold')
            plt.tight_layout()
            figures.append(fig2)
        
        # Plot 3: Data Efficiency
        if efficiency:
            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            methods = list(efficiency.keys())
            acc_savings = [efficiency[m]['avg_accuracy_data_savings_pct'] for m in methods]
            f1_savings = [efficiency[m]['avg_macro_f1_data_savings_pct'] for m in methods]
            
            bars1 = ax1.bar(methods, acc_savings, color='lightgreen', alpha=0.7)
            ax1.set_ylabel('Data Savings (%)')
            ax1.set_title('Training Data Savings for Same Accuracy')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            for bar, value in zip(bars1, acc_savings):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            bars2 = ax2.bar(methods, f1_savings, color='gold', alpha=0.7)
            ax2.set_ylabel('Data Savings (%)')
            ax2.set_title('Training Data Savings for Same Macro F1')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, f1_savings):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            fig3.suptitle('Data Efficiency Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            figures.append(fig3)
        
        # Plot 4: Comparison with Full Labeled Data
        if comparisons:
            fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            methods = list(comparisons.keys())
            acc_gaps = [comparisons[m]['accuracy_gap_to_full_pct'] for m in methods]
            f1_gaps = [comparisons[m]['macro_f1_gap_to_full_pct'] for m in methods]
            
            bars1 = ax1.bar(methods, acc_gaps, color='orange', alpha=0.7)
            ax1.set_ylabel('Performance Gap (%)')
            ax1.set_title('Accuracy Gap to Full Labeled Data')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            for bar, value in zip(bars1, acc_gaps):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            bars2 = ax2.bar(methods, f1_gaps, color='purple', alpha=0.7)
            ax2.set_ylabel('Performance Gap (%)')
            ax2.set_title('Macro F1 Gap to Full Labeled Data')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, f1_gaps):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            fig4.suptitle('Performance Gap to Full Labeled Data', fontsize=16, fontweight='bold')
            plt.tight_layout()
            figures.append(fig4)
        
        return figures
    
    def generate_pdf_report(self, curves: Dict, improvements: Dict, efficiency: Dict, 
                           comparisons: Dict, stats: Dict, output_path: str = 'active_learning_analysis.pdf'):
        """Generate comprehensive PDF report"""
        
        with PdfPages(output_path) as pdf:
            # Create plots
            figures = self.create_plots(curves, improvements, efficiency, comparisons, stats)
            
            # Add plots to PDF
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Create summary text page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Build summary text
            summary_text = self._build_summary_text(curves, improvements, efficiency, comparisons, stats)
            
            # Add text to figure
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"PDF report generated: {output_path}")
    
    def _build_summary_text(self, curves: Dict, improvements: Dict, efficiency: Dict, 
                          comparisons: Dict, stats: Dict) -> str:
        """Build summary text for PDF"""
        
        text = f"""
ACTIVE LEARNING COMPREHENSIVE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}

FULL LABELED DATA REFERENCE:
- Accuracy: {self.full_labeled_data['accuracy']:.3f}
- Macro F1: {self.full_labeled_data['macro_f1']:.3f}
- Training Size: {self.full_labeled_data['train_data_size']}

{'='*80}

PERFORMANCE IMPROVEMENT OVER RANDOM SAMPLING:
"""
        
        if improvements:
            for method, imp in improvements.items():
                text += f"""
{method.replace('_', ' ').title()}:
  Accuracy Improvement: {imp['accuracy_improvement_pct']:.2f}%
  Macro F1 Improvement: {imp['macro_f1_improvement_pct']:.2f}%
  Max Accuracy Improvement: {imp['max_accuracy_improvement_pct']:.2f}%
  Max Macro F1 Improvement: {imp['max_macro_f1_improvement_pct']:.2f}%
"""
        
        text += f"""
{'='*80}

DATA EFFICIENCY ANALYSIS:
"""
        
        if efficiency:
            for method, eff in efficiency.items():
                text += f"""
{method.replace('_', ' ').title()}:
  Avg Accuracy Data Savings: {eff['avg_accuracy_data_savings_pct']:.2f}%
  Avg Macro F1 Data Savings: {eff['avg_macro_f1_data_savings_pct']:.2f}%
  Max Accuracy Data Savings: {eff['max_accuracy_data_savings_pct']:.2f}%
  Max Macro F1 Data Savings: {eff['max_macro_f1_data_savings_pct']:.2f}%
"""
        
        text += f"""
{'='*80}

COMPARISON WITH FULL LABELED DATA:
"""
        
        if comparisons:
            for method, comp in comparisons.items():
                text += f"""
{method.replace('_', ' ').title()}:
  Final Accuracy: {comp['final_accuracy']:.3f}
  Final Macro F1: {comp['final_macro_f1']:.3f}
  Final Training Size: {comp['final_train_size']}
  Accuracy Gap to Full: {comp['accuracy_gap_to_full_pct']:.2f}%
  Macro F1 Gap to Full: {comp['macro_f1_gap_to_full_pct']:.2f}%
  Data Efficiency vs Full: {comp['data_efficiency_vs_full']:.2f}%
"""
        
        text += f"""
{'='*80}

STATISTICAL ANALYSIS:
"""
        
        if stats:
            text += f"""
Methods Analyzed: {', '.join(stats['methods'])}

Accuracy Statistics:
  Mean: {stats['accuracy_stats']['mean']:.3f}
  Std Dev: {stats['accuracy_stats']['std']:.3f}
  Min: {stats['accuracy_stats']['min']:.3f}
  Max: {stats['accuracy_stats']['max']:.3f}

Macro F1 Statistics:
  Mean: {stats['macro_f1_stats']['mean']:.3f}
  Std Dev: {stats['macro_f1_stats']['std']:.3f}
  Min: {stats['macro_f1_stats']['min']:.3f}
  Max: {stats['macro_f1_stats']['max']:.3f}
"""
            
            for key, value in stats.items():
                if key.endswith('_vs_random'):
                    method_name = key.replace('_vs_random', '')
                    text += f"""
{method_name.replace('_', ' ').title()} vs Random Sampling:
  Accuracy Difference: {value['accuracy_difference']:.3f}
  Macro F1 Difference: {value['macro_f1_difference']:.3f}
  Accuracy Improvement: {value['accuracy_improvement_pct']:.2f}%
  Macro F1 Improvement: {value['macro_f1_improvement_pct']:.2f}%
"""
        
        text += f"""
{'='*80}

KEY FINDINGS:
"""
        
        # Add key findings
        if improvements:
            best_method = max(improvements.keys(), 
                            key=lambda x: improvements[x]['accuracy_improvement_pct'])
            best_improvement = improvements[best_method]['accuracy_improvement_pct']
            text += f"""
- Best performing method: {best_method.replace('_', ' ').title()}
- Highest accuracy improvement: {best_improvement:.2f}% over random sampling
"""
        
        if efficiency:
            most_efficient = max(efficiency.keys(), 
                               key=lambda x: efficiency[x]['avg_accuracy_data_savings_pct'])
            best_savings = efficiency[most_efficient]['avg_accuracy_data_savings_pct']
            text += f"""
- Most data efficient: {most_efficient.replace('_', ' ').title()}
- Highest data savings: {best_savings:.2f}% for same accuracy
"""
        
        if comparisons:
            closest_to_full = min(comparisons.keys(), 
                                key=lambda x: comparisons[x]['accuracy_gap_to_full_pct'])
            smallest_gap = comparisons[closest_to_full]['accuracy_gap_to_full_pct']
            text += f"""
- Closest to full labeled performance: {closest_to_full.replace('_', ' ').title()}
- Smallest performance gap: {smallest_gap:.2f}% to full data
"""
        
        return text
    
    def analyze_files(self, file1_path: str, file2_path: str, output_pdf: str = 'active_learning_analysis.pdf'):
        """Main analysis function for two result files"""
        
        print(f"Loading data from {file1_path} and {file2_path}...")
        
        # Load both files
        df1 = self.load_and_parse_data(file1_path)
        df2 = self.load_and_parse_data(file2_path)
        
        # Combine data
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        print(f"Combined dataset shape: {combined_df.shape}")
        print(f"Methods found: {list(combined_df['method'].unique())}")
        
        # Extract curves
        curves = self.extract_performance_curves(combined_df)
        
        # Perform analyses
        print("Calculating performance improvements...")
        improvements = self.calculate_performance_improvement(curves)
        
        print("Calculating data efficiency...")
        efficiency = self.calculate_data_efficiency(curves)
        
        print("Comparing with full labeled data...")
        comparisons = self.compare_with_full_labeled(curves)
        
        print("Performing statistical analysis...")
        stats = self.statistical_analysis(curves)
        
        # Generate report
        print("Generating PDF report...")
        self.generate_pdf_report(curves, improvements, efficiency, comparisons, stats, output_pdf)
        
        # Print summary
        self._print_summary(improvements, efficiency, comparisons)
        
        return {
            'curves': curves,
            'improvements': improvements,
            'efficiency': efficiency,
            'comparisons': comparisons,
            'statistics': stats
        }
    
    def _print_summary(self, improvements: Dict, efficiency: Dict, comparisons: Dict):
        """Print quick summary to console"""
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        
        if improvements:
            print("\nPERFORMANCE IMPROVEMENTS OVER RANDOM SAMPLING:")
            for method, imp in improvements.items():
                print(f"  {method.replace('_', ' ').title():20}: {imp['accuracy_improvement_pct']:+6.2f}% accuracy")
        
        if efficiency:
            print("\nDATA EFFICIENCY (SAVINGS OVER RANDOM):")
            for method, eff in efficiency.items():
                print(f"  {method.replace('_', ' ').title():20}: {eff['avg_accuracy_data_savings_pct']:+6.2f}% data")
        
        if comparisons:
            print("\nCOMPARISON WITH FULL LABELED DATA:")
            for method, comp in comparisons.items():
                print(f"  {method.replace('_', ' ').title():20}: {comp['accuracy_gap_to_full_pct']:+6.2f}% gap")
        
        print("\n" + "="*60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Active Learning Analysis')
    parser.add_argument('--file1', default='results_N1500.xlsx', help='First results file')
    parser.add_argument('--file2', default='results_N300.xlsx', help='Second results file')
    parser.add_argument('--output', default='active_learning_analysis.pdf', help='Output PDF file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.file1):
        print(f"Error: File not found: {args.file1}")
        return
    
    if not os.path.exists(args.file2):
        print(f"Error: File not found: {args.file2}")
        return
    
    # Run analysis
    analyzer = ActiveLearningAnalyzer()
    results = analyzer.analyze_files(args.file1, args.file2, args.output)
    
    print(f"\nAnalysis complete! Report saved to: {args.output}")


if __name__ == "__main__":
    main()
