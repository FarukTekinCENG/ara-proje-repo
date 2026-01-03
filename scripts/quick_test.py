#!/usr/bin/env python3
"""
Quick test script for model evaluation
Simple interface to test models on CSV datasets
"""

import os
import sys
import argparse

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from demo_features import DemoFeatures


def quick_test(csv_path, model_path, text_column="description", label_column="formatted_work_type", 
               sample_size=None, verbose=True):
    """
    Quick test of a model on a CSV dataset
    
    Args:
        csv_path: Path to CSV file
        model_path: Path to trained model
        text_column: Name of text column
        label_column: Name of label column
        sample_size: Optional sample size for faster testing
        verbose: Whether to print detailed output
    
    Returns:
        dict: Test results
    """
    if verbose:
        print(f"Testing model: {model_path}")
        print(f"Dataset: {csv_path}")
    
    # Check if files exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    demo = DemoFeatures(model_path)
    
    # Test on CSV
    accuracy, metrics = demo.evaluate_on_csv(
        csv_path=csv_path,
        text_column=text_column,
        label_column=label_column,
        sample_size=sample_size
    )
    
    if verbose:
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.3f}")
        
        if metrics:
            print(f"Total samples: {metrics.get('total_samples', 'N/A')}")
            print(f"Correct predictions: {metrics.get('correct_predictions', 'N/A')}")
            
            if 'per_class_accuracy' in metrics:
                print(f"\nPer-class accuracy:")
                for label, acc in sorted(metrics['per_class_accuracy'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {label}: {acc:.3f}")
    
    return {
        "accuracy": accuracy,
        "metrics": metrics,
        "model_path": model_path,
        "csv_path": csv_path
    }


def interactive_test():
    """Interactive mode for testing"""
    print("=== INTERACTIVE MODEL TEST ===\n")
    
    # Get CSV path
    csv_path = input("Enter CSV path (default: data/temiz_etiketli_ilanlar_v5.csv): ").strip()
    if not csv_path:
        csv_path = "data/temiz_etiketli_ilanlar_v5.csv"
    
    # Get model path
    model_path = input("Enter model path (default: trained_models/model1): ").strip()
    if not model_path:
        model_path = "trained_models/model1"
    
    # Get sample size (optional)
    sample_input = input("Enter sample size (leave empty for full dataset): ").strip()
    sample_size = None
    if sample_input:
        try:
            sample_size = int(sample_input)
        except ValueError:
            print("Invalid sample size, using full dataset")
    
    # Run test
    try:
        results = quick_test(csv_path, model_path, sample_size=sample_size)
        print(f"\nTest completed successfully!")
        print(f"Accuracy: {results['accuracy']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")


def batch_test():
    """Batch test multiple models"""
    print("=== BATCH MODEL TEST ===\n")
    
    # Get CSV path
    csv_path = input("Enter CSV path (default: data/temiz_etiketli_ilanlar_v5.csv): ").strip()
    if not csv_path:
        csv_path = "data/temiz_etiketli_ilanlar_v5.csv"
    
    # Find models
    model_dir = "trained_models"
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return
    
    models = []
    for item in os.listdir(model_dir):
        model_path = os.path.join(model_dir, item)
        if os.path.isdir(model_path) and item.startswith("model"):
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                models.append((item, model_path))
    
    if not models:
        print("No valid models found")
        return
    
    print(f"Found {len(models)} models")
    
    # Get sample size
    sample_input = input("Enter sample size per model (leave empty for full dataset): ").strip()
    sample_size = None
    if sample_input:
        try:
            sample_size = int(sample_input)
        except ValueError:
            print("Invalid sample size, using full dataset")
    
    # Test each model
    results = {}
    for model_name, model_path in models:
        print(f"\nTesting {model_name}...")
        try:
            result = quick_test(csv_path, model_path, sample_size=sample_size, verbose=False)
            results[model_name] = result
            print(f"  Accuracy: {result['accuracy']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {"error": str(e)}
    
    # Summary
    print(f"\n=== BATCH TEST RESULTS ===")
    valid_results = {name: data for name, data in results.items() if "accuracy" in data}
    
    if valid_results:
        ranking = sorted(valid_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for i, (model_name, result) in enumerate(ranking, 1):
            print(f"{i}. {model_name}: {result['accuracy']:.3f}")
        
        best_model = ranking[0][0]
        print(f"\nBest model: {best_model}")
    else:
        print("No successful tests")


def main():
    parser = argparse.ArgumentParser(description="Quick model testing tool")
    parser.add_argument("--csv", type=str, help="CSV file path")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--sample", type=int, help="Sample size for testing")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", action="store_true", help="Batch test multiple models")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test()
    elif args.batch:
        batch_test()
    else:
        # Command line mode
        csv_path = args.csv or "data/temiz_etiketli_ilanlar_v5.csv"
        model_path = args.model or "trained_models/model1"
        
        try:
            results = quick_test(csv_path, model_path, sample_size=args.sample)
            print(f"Test completed: {results['accuracy']:.3f} accuracy")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
