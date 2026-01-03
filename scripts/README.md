# Model Evaluation Scripts

This directory contains scripts for comprehensive model evaluation without modifying the core active learning code.

## Scripts Overview

### 1. `quick_test.py` - Quick Model Testing
**Purpose**: Simple and fast model testing on CSV datasets

**Usage Examples**:
```bash
# Basic usage
python scripts/quick_test.py --csv data/temiz_etiketli_ilanlar_v5.csv --model trained_models/model1

# With sample size for faster testing
python scripts/quick_test.py --csv data/temiz_etiketli_ilanlar_v5.csv --model trained_models/model1 --sample 1000

# Interactive mode
python scripts/quick_test.py --interactive

# Batch test all models
python scripts/quick_test.py --batch
```

**Features**:
- Single model testing
- Batch testing of multiple models
- Interactive mode
- Sample size control for faster testing
- Per-class accuracy reporting

---

### 2. `cross_validation.py` - Cross-Validation
**Purpose**: Robust model evaluation using k-fold cross-validation

**Usage Examples**:
```bash
# 5-fold cross-validation
python scripts/cross_validation.py

# Custom parameters (edit the script or import as module)
from scripts.cross_validation import cross_validate_model
results = cross_validate_model(
    csv_path="data/temiz_etiketli_ilanlar_v5.csv",
    model_path="trained_models/model1",
    k_folds=5,
    stratified=True
)
```

**Features**:
- K-fold cross-validation
- Stratified sampling option
- Per-fold accuracy tracking
- Statistical analysis (mean ± std)
- JSON result export

---

### 3. `model_comparison.py` - Model Comparison
**Purpose**: Compare multiple trained models on the same dataset

**Usage Examples**:
```bash
# Compare all models in trained_models directory
python scripts/model_comparison.py

# As module usage
from scripts.model_comparison import compare_models
results = compare_models(
    csv_path="data/temizetikli_ilanlar_v5.csv",
    model_dir="trained_models",
    sample_size=1000
)
```

**Features**:
- Multi-model comparison
- Ranking by accuracy
- Per-class performance analysis
- Confidence analysis
- Human-readable reports
- JSON and text output formats

---

### 4. `error_analysis.py` - Error Analysis
**Purpose**: Deep analysis of prediction errors and patterns

**Usage Examples**:
```bash
# Analyze model errors
python scripts/error_analysis.py

# As module usage
from scripts.error_analysis import analyze_errors
results = analyze_errors(
    csv_path="data/temiz_etiketli_ilanlar_v5.csv",
    model_path="trained_models/model1",
    max_errors=500
)
```

**Features**:
- Detailed error analysis
- Confusion pattern identification
- Confidence analysis for errors vs correct predictions
- Error example extraction
- High-confidence error detection
- CSV export of all errors

---

## Quick Start Guide

### 1. Test Your Model
```bash
# Quick test of model1
python scripts/quick_test.py --model trained_models/model1
```

### 2. Compare All Models
```bash
# See which model performs best
python scripts/model_comparison.py
```

### 3. Analyze Errors
```bash
# Understand what your model gets wrong
python scripts/error_analysis.py --model trained_models/model1
```

### 4. Robust Evaluation
```bash
# Cross-validation for reliable results
python scripts/cross_validation.py
```

## Output Files

Each script generates output files:

- `cross_validation_results.json` - Cross-validation metrics
- `model_comparison_results.json` - Model comparison data
- `model_comparison_report.txt` - Human-readable comparison
- `error_analysis_results.json` - Detailed error analysis
- `error_analysis_report.txt` - Human-readable error report
- `error_analysis_*.csv` - CSV file with all prediction errors

## CSV Format Requirements

The scripts expect CSV files with:
- `description` column (job posting text)
- `formatted_work_type` column (job type labels)

If your CSV has different column names, you can specify them:
```python
# Example usage with custom column names
results = quick_test(
    csv_path="your_data.csv",
    model_path="trained_models/model1",
    text_column="job_text",      # Custom text column
    label_column="job_category"   # Custom label column
)
```

## Integration with Existing Code

These scripts use the existing `DemoFeatures` class without any modifications to the core active learning system. They can be run independently or imported as modules in your own evaluation pipelines.

## Performance Tips

1. **Use sample sizes** for large datasets to speed up testing
2. **Batch testing** is faster than individual tests for multiple models
3. **Error analysis** can be memory-intensive for large datasets - use `max_errors` parameter
4. **Cross-validation** is computationally expensive - consider smaller k_folds for quick results

## Dependencies

All scripts use only the existing dependencies:
- `pandas` for data handling
- `numpy` for numerical operations  
- `sklearn` for cross-validation
- The existing `DemoFeatures` class

No additional packages need to be installed.
