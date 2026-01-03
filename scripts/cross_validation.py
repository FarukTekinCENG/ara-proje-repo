#!/usr/bin/env python3
"""
Cross-validation script for model evaluation
Tests a single model across multiple folds of the dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from demo_features import DemoFeatures


def cross_validate_model(csv_path, model_path, k_folds=5, stratified=True, text_column="description", label_column="formatted_work_type"):
    """
    Perform cross-validation on a single model
    
    Args:
        csv_path: Path to CSV file
        model_path: Path to trained model
        k_folds: Number of folds
        stratified: Whether to use stratified sampling
        text_column: Name of text column
        label_column: Name of label column
    
    Returns:
        dict: Cross-validation results
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {dict(Counter(df[label_column]))}")
    
    # Create cross-validation splits
    if stratified:
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = kf.split(df, df[label_column])
    else:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = kf.split(df)
    
    results = {
        "fold_accuracies": [],
        "fold_metrics": [],
        "mean_accuracy": None,
        "std_accuracy": None,
        "model_path": model_path,
        "csv_path": csv_path,
        "k_folds": k_folds,
        "stratified": stratified
    }
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")
        
        # Test seti
        test_df = df.iloc[test_idx].copy()
        print(f"Test set size: {len(test_df)}")
        
        try:
            # DemoFeatures ile test
            demo = DemoFeatures(model_path)
            
            # Batch prediction
            texts = test_df[text_column].tolist()
            predictions = []
            confidences = []
            
            print("Making predictions...")
            for text in texts:
                result = demo.predict_single(text)
                predictions.append(result["predicted_class"])
                confidences.append(result["confidence"])
            
            # Calculate accuracy
            true_labels = test_df[label_column].values
            correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
            accuracy = correct / len(predictions)
            
            # Calculate per-class metrics
            label_counts = Counter(true_labels)
            correct_by_class = {}
            for label in label_counts:
                label_mask = true_labels == label
                label_correct = sum(1 for true, pred in zip(true_labels, predictions) 
                                  if true == pred and true == label)
                correct_by_class[label] = label_correct / sum(label_mask)
            
            fold_metrics = {
                "accuracy": accuracy,
                "per_class_accuracy": correct_by_class,
                "test_size": len(test_df),
                "mean_confidence": np.mean(confidences)
            }
            
            results["fold_accuracies"].append(accuracy)
            results["fold_metrics"].append(fold_metrics)
            
            print(f"Fold {fold + 1} accuracy: {accuracy:.3f}")
            print(f"Mean confidence: {np.mean(confidences):.3f}")
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            results["fold_accuracies"].append(0.0)
            results["fold_metrics"].append({"error": str(e)})
    
    # Calculate summary statistics
    results["mean_accuracy"] = np.mean(results["fold_accuracies"])
    results["std_accuracy"] = np.std(results["fold_accuracies"])
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Mean accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    print(f"Fold accuracies: {[f'{acc:.3f}' for acc in results['fold_accuracies']]}")
    
    return results


def save_results(results, output_path="cross_validation_results.json"):
    """Save cross-validation results to JSON file"""
    import json
    
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            serializable_results[key] = float(value)
        elif isinstance(value, dict):
            serializable_results[key] = {k: (float(v) if isinstance(v, (np.int64, np.float64)) else v) 
                                        for k, v in value.items()}
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


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
    
    # Run cross-validation
    results = cross_validate_model(
        csv_path=csv_path,
        model_path=model_path,
        k_folds=5,
        stratified=True
    )
    
    # Save results
    save_results(results)
