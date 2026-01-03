#!/usr/bin/env python3
"""
Model comparison script
Compares multiple trained models on the same test dataset
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from collections import Counter
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from demo_features import DemoFeatures


def compare_models(csv_path, model_dir="trained_models", text_column="description", label_column="formatted_work_type", sample_size=None):
    """
    Compare multiple models on the same dataset
    
    Args:
        csv_path: Path to CSV file
        model_dir: Directory containing trained models
        text_column: Name of text column
        label_column: Name of label column
        sample_size: Optional sample size for faster testing
    
    Returns:
        dict: Comparison results
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Optional sampling for faster testing
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} examples from dataset")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {dict(Counter(df[label_column]))}")
    
    # Find available models
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")
    
    models = []
    for item in os.listdir(model_dir):
        model_path = os.path.join(model_dir, item)
        if os.path.isdir(model_path) and item.startswith("model"):
            # Check if it's a valid HuggingFace model
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                models.append((item, model_path))
    
    if not models:
        raise ValueError(f"No valid models found in {model_dir}")
    
    print(f"Found {len(models)} models: {[name for name, _ in models]}")
    
    results = {
        "comparison_timestamp": datetime.now().isoformat(),
        "csv_path": csv_path,
        "model_dir": model_dir,
        "dataset_size": len(df),
        "sample_size": sample_size,
        "models": {},
        "ranking": []
    }
    
    # Test each model
    for model_name, model_path in models:
        print(f"\n=== Testing {model_name} ===")
        
        try:
            demo = DemoFeatures(model_path)
            
            # Batch prediction
            texts = df[text_column].tolist()
            predictions = []
            confidences = []
            
            print("Making predictions...")
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(texts)}")
                
                result = demo.predict_single(text)
                predictions.append(result["predicted_class"])
                confidences.append(result["confidence"])
            
            # Calculate metrics
            true_labels = df[label_column].values
            correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
            accuracy = correct / len(predictions)
            
            # Per-class accuracy
            label_counts = Counter(true_labels)
            per_class_accuracy = {}
            for label in label_counts:
                label_mask = true_labels == label
                label_correct = sum(1 for true, pred in zip(true_labels, predictions) 
                                  if true == pred and true == label)
                per_class_accuracy[label] = label_correct / sum(label_mask)
            
            # Additional metrics
            mean_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            model_results = {
                "accuracy": accuracy,
                "per_class_accuracy": per_class_accuracy,
                "mean_confidence": mean_confidence,
                "confidence_std": confidence_std,
                "total_predictions": len(predictions),
                "correct_predictions": correct,
                "model_path": model_path
            }
            
            results["models"][model_name] = model_results
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Mean confidence: {mean_confidence:.3f} ± {confidence_std:.3f}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            results["models"][model_name] = {
                "error": str(e),
                "model_path": model_path
            }
    
    # Rank models by accuracy
    valid_models = {name: data for name, data in results["models"].items() if "accuracy" in data}
    ranking = sorted(valid_models.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    results["ranking"] = [{"model": name, "accuracy": data["accuracy"]} for name, data in ranking]
    
    # Print summary
    print(f"\n=== Model Ranking ===")
    for i, (model_name, data) in enumerate(ranking, 1):
        print(f"{i}. {model_name}: {data['accuracy']:.3f}")
    
    if ranking:
        best_model = ranking[0][0]
        best_accuracy = ranking[0][1]["accuracy"]
        print(f"\nBest model: {best_model} with accuracy {best_accuracy:.3f}")
        
        # Show best model's per-class performance
        best_metrics = results["models"][best_model]
        if "per_class_accuracy" in best_metrics:
            print(f"\nBest model per-class accuracy:")
            for label, acc in sorted(best_metrics["per_class_accuracy"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {acc:.3f}")
    
    return results


def save_comparison_results(results, output_path="model_comparison_results.json"):
    """Save comparison results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Comparison results saved to {output_path}")


def generate_comparison_report(results, output_path="model_comparison_report.txt"):
    """Generate a human-readable comparison report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {results['csv_path']}\n")
        f.write(f"Dataset size: {results['dataset_size']}\n")
        if results['sample_size']:
            f.write(f"Sample size: {results['sample_size']}\n")
        f.write(f"Models tested: {len(results['models'])}\n")
        f.write(f"Test date: {results['comparison_timestamp']}\n\n")
        
        f.write("RANKING:\n")
        f.write("-" * 20 + "\n")
        for i, model_data in enumerate(results['ranking'], 1):
            f.write(f"{i}. {model_data['model']}: {model_data['accuracy']:.3f}\n")
        
        f.write("\nDETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for model_name, model_data in results['models'].items():
            f.write(f"\n{model_name}:\n")
            if "error" in model_data:
                f.write(f"  ERROR: {model_data['error']}\n")
            else:
                f.write(f"  Accuracy: {model_data['accuracy']:.3f}\n")
                f.write(f"  Mean confidence: {model_data['mean_confidence']:.3f} ± {model_data['confidence_std']:.3f}\n")
                f.write(f"  Correct/Total: {model_data['correct_predictions']}/{model_data['total_predictions']}\n")
                
                if "per_class_accuracy" in model_data:
                    f.write("  Per-class accuracy:\n")
                    for label, acc in sorted(model_data["per_class_accuracy"].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"    {label}: {acc:.3f}\n")
    
    print(f"Comparison report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    csv_path = "data/temiz_etiketli_ilanlar_v5.csv"
    model_dir = "trained_models"
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Run comparison
    results = compare_models(
        csv_path=csv_path,
        model_dir=model_dir,
        sample_size=1000  # Optional: limit for faster testing
    )
    
    # Save results
    save_comparison_results(results)
    generate_comparison_report(results)
