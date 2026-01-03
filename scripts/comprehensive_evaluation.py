#!/usr/bin/env python3
"""
Comprehensive model evaluation runner
Combines all evaluation scripts into one comprehensive analysis
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from scripts.cross_validation import cross_validate_model, save_results as save_cv_results
from scripts.model_comparison import compare_models, save_comparison_results, generate_comparison_report
from scripts.error_analysis import analyze_errors, save_error_analysis, generate_error_report
from scripts.quick_test import quick_test


def comprehensive_evaluation(csv_path, model_dir="trained_models", target_model=None, 
                           sample_size=None, cv_folds=3, max_errors=500):
    """
    Run comprehensive evaluation combining all analysis methods
    
    Args:
        csv_path: Path to CSV file
        model_dir: Directory containing trained models
        target_model: Specific model to evaluate (None for all models)
        sample_size: Sample size for faster testing
        cv_folds: Number of cross-validation folds
        max_errors: Maximum errors to analyze
    
    Returns:
        dict: Comprehensive evaluation results
    """
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    print(f"Dataset: {csv_path}")
    print(f"Model directory: {model_dir}")
    print(f"Target model: {target_model or 'All models'}")
    print(f"Sample size: {sample_size or 'Full dataset'}")
    print(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "csv_path": csv_path,
        "model_dir": model_dir,
        "target_model": target_model,
        "sample_size": sample_size,
        "quick_test": {},
        "model_comparison": {},
        "cross_validation": {},
        "error_analysis": {}
    }
    
    # 1. Quick Test
    print("\n1. QUICK TEST")
    print("-" * 30)
    
    if target_model:
        model_path = os.path.join(model_dir, target_model)
        if os.path.exists(model_path):
            try:
                quick_results = quick_test(csv_path, model_path, sample_size=sample_size, verbose=False)
                results["quick_test"] = quick_results
                print(f"✓ {target_model}: {quick_results['accuracy']:.3f} accuracy")
            except Exception as e:
                print(f"✗ Quick test failed: {e}")
                results["quick_test"]["error"] = str(e)
        else:
            print(f"✗ Model not found: {model_path}")
    else:
        print("Skipping quick test (no target model specified)")
    
    # 2. Model Comparison
    print("\n2. MODEL COMPARISON")
    print("-" * 30)
    
    try:
        comparison_results = compare_models(csv_path, model_dir, sample_size=sample_size)
        results["model_comparison"] = comparison_results
        
        if comparison_results.get("ranking"):
            print(f"✓ Compared {len(comparison_results['ranking'])} models")
            best_model = comparison_results["ranking"][0]
            print(f"✓ Best model: {best_model['model']} ({best_model['accuracy']:.3f})")
        else:
            print("✗ No valid models found")
    except Exception as e:
        print(f"✗ Model comparison failed: {e}")
        results["model_comparison"]["error"] = str(e)
    
    # 3. Cross-Validation (on best model or target model)
    print("\n3. CROSS-VALIDATION")
    print("-" * 30)
    
    cv_model_path = None
    if target_model:
        cv_model_path = os.path.join(model_dir, target_model)
    elif results["model_comparison"].get("ranking"):
        best_model_name = results["model_comparison"]["ranking"][0]["model"]
        cv_model_path = os.path.join(model_dir, best_model_name)
    
    if cv_model_path and os.path.exists(cv_model_path):
        try:
            cv_results = cross_validate_model(
                csv_path, cv_model_path, k_folds=cv_folds, stratified=True
            )
            results["cross_validation"] = cv_results
            print(f"✓ {cv_folds}-fold CV: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        except Exception as e:
            print(f"✗ Cross-validation failed: {e}")
            results["cross_validation"]["error"] = str(e)
    else:
        print("Skipping cross-validation (no suitable model found)")
    
    # 4. Error Analysis (on best model or target model)
    print("\n4. ERROR ANALYSIS")
    print("-" * 30)
    
    error_model_path = cv_model_path  # Use same model as CV
    if error_model_path and os.path.exists(error_model_path):
        try:
            error_results = analyze_errors(
                csv_path, error_model_path, max_errors=max_errors
            )
            results["error_analysis"] = error_results
            print(f"✓ Analyzed {error_results['error_predictions']} errors")
            print(f"✓ Error rate: {error_results['error_rate']:.3f}")
        except Exception as e:
            print(f"✗ Error analysis failed: {e}")
            results["error_analysis"]["error"] = str(e)
    else:
        print("Skipping error analysis (no suitable model found)")
    
    # 5. Summary
    print("\n5. EVALUATION SUMMARY")
    print("-" * 30)
    
    summary_lines = []
    
    if results["quick_test"].get("accuracy"):
        acc = results["quick_test"]["accuracy"]
        summary_lines.append(f"Quick test accuracy: {acc:.3f}")
    
    if results["model_comparison"].get("ranking"):
        best = results["model_comparison"]["ranking"][0]
        summary_lines.append(f"Best model: {best['model']} ({best['accuracy']:.3f})")
    
    if results["cross_validation"].get("mean_accuracy"):
        mean_acc = results["cross_validation"]["mean_accuracy"]
        std_acc = results["cross_validation"]["std_accuracy"]
        summary_lines.append(f"Cross-validation: {mean_acc:.3f} ± {std_acc:.3f}")
    
    if results["error_analysis"].get("error_rate"):
        error_rate = results["error_analysis"]["error_rate"]
        summary_lines.append(f"Error rate: {error_rate:.3f}")
    
    if summary_lines:
        for line in summary_lines:
            print(f"• {line}")
    else:
        print("No successful evaluations completed")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    if results["model_comparison"]:
        save_comparison_results(results["model_comparison"], 
                               os.path.join(output_dir, "model_comparison.json"))
        generate_comparison_report(results["model_comparison"], 
                                  os.path.join(output_dir, "model_comparison_report.txt"))
    
    if results["cross_validation"]:
        save_cv_results(results["cross_validation"], 
                       os.path.join(output_dir, "cross_validation.json"))
    
    if results["error_analysis"]:
        save_error_analysis(results["error_analysis"], 
                           os.path.join(output_dir, "error_analysis.json"))
        generate_error_report(results["error_analysis"], 
                            os.path.join(output_dir, "error_analysis_report.txt"))
    
    # Save comprehensive summary
    import json
    with open(os.path.join(output_dir, "comprehensive_evaluation.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ All results saved to: {output_dir}/")
    
    return results, output_dir


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--csv", type=str, default="data/temiz_etiketli_ilanlar_v5.csv", 
                       help="CSV file path")
    parser.add_argument("--model-dir", type=str, default="trained_models", 
                       help="Directory containing trained models")
    parser.add_argument("--target-model", type=str, 
                       help="Specific model to evaluate (default: all models)")
    parser.add_argument("--sample", type=int, 
                       help="Sample size for faster testing")
    parser.add_argument("--cv-folds", type=int, default=3, 
                       help="Number of cross-validation folds")
    parser.add_argument("--max-errors", type=int, default=500, 
                       help="Maximum errors to analyze")
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    try:
        results, output_dir = comprehensive_evaluation(
            csv_path=args.csv,
            model_dir=args.model_dir,
            target_model=args.target_model,
            sample_size=args.sample,
            cv_folds=args.cv_folds,
            max_errors=args.max_errors
        )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
