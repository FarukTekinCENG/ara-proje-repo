#!/usr/bin/env python3
"""
Error analysis script for model evaluation
Analyzes prediction errors and provides detailed insights
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from collections import Counter, defaultdict
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from demo_features import DemoFeatures


def analyze_errors(csv_path, model_path, text_column="description", label_column="formatted_work_type", 
                  save_errors=True, max_errors=1000):
    """
    Analyze prediction errors in detail
    
    Args:
        csv_path: Path to CSV file
        model_path: Path to trained model
        text_column: Name of text column
        label_column: Name of label column
        save_errors: Whether to save error examples to file
        max_errors: Maximum number of errors to analyze (for large datasets)
    
    Returns:
        dict: Error analysis results
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Optional sampling for large datasets
    if max_errors and len(df) > max_errors * 2:  # Keep enough for good error analysis
        df = df.sample(n=max_errors * 2, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} examples for analysis")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {dict(Counter(df[label_column]))}")
    
    print(f"Loading model from {model_path}...")
    demo = DemoFeatures(model_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    confidences = []
    
    for i, text in enumerate(df[text_column]):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(df)}")
        
        result = demo.predict_single(text)
        predictions.append(result["predicted_class"])
        confidences.append(result["confidence"])
    
    # Analyze errors
    df["predicted"] = predictions
    df["confidence"] = confidences
    
    # Find errors
    errors = df[df[label_column] != df["predicted"]].copy()
    correct = df[df[label_column] == df["predicted"]].copy()
    
    total_accuracy = len(correct) / len(df)
    error_rate = len(errors) / len(df)
    
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Total samples: {len(df)}")
    print(f"Correct predictions: {len(correct)} ({total_accuracy:.3f})")
    print(f"Errors: {len(errors)} ({error_rate:.3f})")
    
    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "csv_path": csv_path,
        "model_path": model_path,
        "total_samples": len(df),
        "correct_predictions": len(correct),
        "error_predictions": len(errors),
        "accuracy": total_accuracy,
        "error_rate": error_rate,
        "error_analysis": {},
        "confidence_analysis": {},
        "confusion_matrix": {},
        "error_examples": []
    }
    
    # Error analysis by true label
    print(f"\n=== ERROR ANALYSIS BY TRUE LABEL ===")
    true_label_errors = errors[label_column].value_counts()
    true_label_totals = df[label_column].value_counts()
    
    error_by_true_label = {}
    for label in true_label_totals.index:
        total = true_label_totals[label]
        error_count = true_label_errors.get(label, 0)
        error_rate = error_count / total
        error_by_true_label[label] = {
            "total": total,
            "errors": error_count,
            "error_rate": error_rate,
            "accuracy": 1 - error_rate
        }
        print(f"{label}: {error_count}/{total} errors ({error_rate:.3f})")
    
    results["error_analysis"]["by_true_label"] = error_by_true_label
    
    # Error analysis by predicted label
    print(f"\n=== ERROR ANALYSIS BY PREDICTED LABEL ===")
    pred_label_errors = errors["predicted"].value_counts()
    
    error_by_pred_label = {}
    for label in pred_label_errors.index:
        error_count = pred_label_errors[label]
        error_by_pred_label[label] = {
            "error_count": error_count,
            "error_percentage": error_count / len(errors)
        }
        print(f"{label}: {error_count} errors ({error_count/len(errors):.3f})")
    
    results["error_analysis"]["by_predicted_label"] = error_by_pred_label
    
    # Confusion matrix for errors
    print(f"\n=== CONFUSION PATTERNS ===")
    confusion_patterns = defaultdict(int)
    for _, row in errors.iterrows():
        true_label = row[label_column]
        pred_label = row["predicted"]
        confusion_patterns[(true_label, pred_label)] += 1
    
    # Top confusion patterns
    top_confusions = sorted(confusion_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    confusion_matrix = {}
    for (true_label, pred_label), count in top_confusions:
        confusion_matrix[f"{true_label} → {pred_label}"] = count
        print(f"{true_label} → {pred_label}: {count} times")
    
    results["confusion_matrix"]["top_patterns"] = confusion_matrix
    
    # Confidence analysis
    print(f"\n=== CONFIDENCE ANALYSIS ===")
    correct_confidences = correct["confidence"].values
    error_confidences = errors["confidence"].values
    
    confidence_stats = {
        "correct": {
            "mean": np.mean(correct_confidences),
            "std": np.std(correct_confidences),
            "min": np.min(correct_confidences),
            "max": np.max(correct_confidences)
        },
        "errors": {
            "mean": np.mean(error_confidences),
            "std": np.std(error_confidences),
            "min": np.min(error_confidences),
            "max": np.max(error_confidences)
        }
    }
    
    print(f"Correct predictions - Mean confidence: {confidence_stats['correct']['mean']:.3f}")
    print(f"Error predictions - Mean confidence: {confidence_stats['errors']['mean']:.3f}")
    
    results["confidence_analysis"] = confidence_stats
    
    # High confidence errors
    high_conf_threshold = 0.8
    high_conf_errors = errors[errors["confidence"] >= high_conf_threshold]
    print(f"\nHigh confidence errors (≥{high_conf_threshold}): {len(high_conf_errors)}")
    
    # Error examples
    print(f"\n=== ERROR EXAMPLES ===")
    error_examples = []
    
    # Sample errors by different patterns
    for (true_label, pred_label), count in top_confusions[:5]:
        pattern_errors = errors[(errors[label_column] == true_label) & 
                               (errors["predicted"] == pred_label)]
        
        if len(pattern_errors) > 0:
            example = pattern_errors.iloc[0]
            error_example = {
                "true_label": example[label_column],
                "predicted_label": example["predicted"],
                "confidence": example["confidence"],
                "text": example[text_column][:200] + "..." if len(example[text_column]) > 200 else example[text_column],
                "pattern_count": count
            }
            error_examples.append(error_example)
            
            print(f"\nPattern: {true_label} → {pred_label} (occurs {count} times)")
            print(f"Confidence: {example['confidence']:.3f}")
            print(f"Text: {example['text']}")
    
    results["error_examples"] = error_examples
    
    # Save errors to file if requested
    if save_errors:
        error_output_path = f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        errors.to_csv(error_output_path, index=False)
        print(f"\nAll errors saved to {error_output_path}")
        results["error_file"] = error_output_path
    
    return results


def save_error_analysis(results, output_path="error_analysis_results.json"):
    """Save error analysis results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Error analysis results saved to {output_path}")


def generate_error_report(results, output_path="error_analysis_report.txt"):
    """Generate a human-readable error analysis report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {results['csv_path']}\n")
        f.write(f"Model: {results['model_path']}\n")
        f.write(f"Analysis date: {results['analysis_timestamp']}\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Accuracy: {results['accuracy']:.3f}\n")
        f.write(f"Error rate: {results['error_rate']:.3f}\n\n")
        
        f.write("ERROR ANALYSIS BY TRUE LABEL:\n")
        f.write("-" * 40 + "\n")
        for label, stats in results['error_analysis']['by_true_label'].items():
            f.write(f"{label}:\n")
            f.write(f"  Total: {stats['total']}\n")
            f.write(f"  Errors: {stats['errors']} ({stats['error_rate']:.3f})\n")
            f.write(f"  Accuracy: {stats['accuracy']:.3f}\n")
        
        f.write("\nTOP CONFUSION PATTERNS:\n")
        f.write("-" * 40 + "\n")
        for pattern, count in results['confusion_matrix']['top_patterns'].items():
            f.write(f"{pattern}: {count} times\n")
        
        f.write("\nCONFIDENCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Correct predictions:\n")
        f.write(f"  Mean confidence: {results['confidence_analysis']['correct']['mean']:.3f}\n")
        f.write(f"  Std: {results['confidence_analysis']['correct']['std']:.3f}\n")
        f.write(f"Error predictions:\n")
        f.write(f"  Mean confidence: {results['confidence_analysis']['errors']['mean']:.3f}\n")
        f.write(f"  Std: {results['confidence_analysis']['errors']['std']:.3f}\n")
        
        f.write("\nERROR EXAMPLES:\n")
        f.write("-" * 30 + "\n")
        for i, example in enumerate(results['error_examples'], 1):
            f.write(f"\nExample {i}:\n")
            f.write(f"  True: {example['true_label']}\n")
            f.write(f"  Predicted: {example['predicted_label']}\n")
            f.write(f"  Confidence: {example['confidence']:.3f}\n")
            f.write(f"  Pattern occurs: {example['pattern_count']} times\n")
            f.write(f"  Text: {example['text']}\n")
    
    print(f"Error analysis report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    csv_path = "data/temiz_etiketli_ilanlar_v5.csv"
    model_path = "trained_models/model1"
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Run error analysis
    results = analyze_errors(
        csv_path=csv_path,
        model_path=model_path,
        max_errors=500  # Limit for faster analysis
    )
    
    # Save results
    save_error_analysis(results)
    generate_error_report(results)
