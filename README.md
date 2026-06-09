## TL;DR

Built an Active Learning system for job posting classification on ~124K real-world samples.

- Reduced labeling cost via uncertainty, diversity, and committee-based sampling
- Maintained strong classification performance under severe class imbalance
- Designed full ML pipeline with iterative retraining + evaluation framework
- Built reproducible experiments across multiple sampling strategies

# Active Learning for Job Posting Classification

> A machine learning system implementing Active Learning strategies to reduce labeling cost while maintaining strong classification performance on job posting data.

---

## 🚀 Overview

This project builds a full **Active Learning experimentation framework** for job posting classification.

Instead of labeling the entire dataset, the system iteratively selects the most informative samples from an unlabeled pool and retrains the model.

The goal is to evaluate how different sampling strategies impact:

- Label efficiency
- Model performance
- Class imbalance behavior
- Training stability across iterations

---

## 🎯 Problem Statement

Large-scale job posting datasets contain hundreds of thousands of unlabeled entries.

Fully labeling these datasets is expensive and often impractical.

This project investigates:

> How effectively can Active Learning reduce labeling cost while preserving classification performance?

---

## 📊 Dataset

This project uses the LinkedIn Job Listings dataset from DataStax:

https://huggingface.co/datasets/datastax/linkedin_job_listings

The dataset is a large-scale tabular + text dataset for real-world NLP classification on job postings.

### Dataset Characteristics

| Property        | Value                                |
|----------------|--------------------------------------|
| Total samples  | ~124,000                             |
| Modalities     | Structured + Unstructured Text       |
| Main text field| description                          |
| Task           | Employment type classification       |

### Key Features

- `title`: Job title  
- `description`: Main job posting text (primary NLP input)  
- `formatted_work_type`: Target label (FULL_TIME, CONTRACT, PART_TIME, etc.)  
- `formatted_experience_level`: Experience level  
- `location`: Job location  
- `company_name`: Employer identity  
- Salary-related fields: partially missing / noisy  

---

## 🧠 Active Learning Strategies

### 1. Uncertainty Sampling
Selects samples where the model has lowest confidence.

### 2. Diversity Sampling
Uses embedding clustering (KMeans) to ensure coverage across data space.

### 3. Query by Committee (QBC)
Multiple models vote; high disagreement samples are selected.

### 4. Random Sampling (Baseline)
Uniform random selection used as a performance baseline.

---

## 🏗️ Model Architecture

### Primary Model
- DistilBERT-base-uncased
- 66M parameters
- 512 token limit
- Fast baseline performance

### Extended Model
- EuroBERT (210M parameters)
- 8192 token context length
- Used for long-context experiments

---

## 🔁 System Pipeline

```
Unlabeled Pool
      ↓
Model Inference
      ↓
Active Learning Strategy
      ↓
Sample Selection
      ↓
Human Annotation
      ↓
Dataset Update
      ↓
Model Retraining
      ↓
Repeat until labeling budget is exhausted
```

---

## 📁 Project Structure

### Core Components

```
train.py          → Training entry point
model.py          → Model definitions
methods/          → Active Learning strategies
utils/            → Metrics, DB, preprocessing utilities
table_def/        → Database schema
```

### Experimentation Layer

```
scripts/
├── active_learning_analysis.py
├── comprehensive_analysis.py
├── comprehensive_evaluation.py
├── model_comparison.py
├── cross_validation.py
├── efficiency_analysis.py
├── error_analysis.py
├── statistical_analysis.py
├── plot_class_recalls.py
├── plot_combined_graphs.py
├── split_pool.py
├── prepare_balanced_dataset.py
└── quick_test.py
```

### Temporary / Experimental Code

```
tmp/
├── diversity_sampling.py
├── extract_portion.py
├── jsonl_to_csv.py
└── Semantic_Similarity.txt
```

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score (Macro)  
- Minority Class Recall  

Macro-F1 and minority recall are emphasized due to class imbalance.

---

## ⚙️ Tech Stack

- PyTorch  
- HuggingFace Transformers  
- scikit-learn  
- SentenceTransformers  
- Pandas  
- NumPy  
- PostgreSQL (schema + optional backend)  

---

## 🛠️ Setup & Installation

### 1. Clone repository
```bash
git clone <repo-url>
cd ara-proje-repo
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

Create `.env` file:

```bash
echo -e "DB_HOST=localhost\nDB_NAME=postgres\nDB_USER=postgres\nDB_PASSWORD=postgres\nDB_PORT=5432\nNEON_API_KEY=YOUR_API_KEY" > .env
```

---

## 🗄️ Database Setup (Optional)

```bash
chmod +x table_def/bootstrap.sh
./table_def/bootstrap.sh
```

---

## 📦 Dataset Preparation

### Split dataset (DB pipeline)
```bash
python -m scripts.split_pool --fraction 0.2 --seed 42 --yes
```

### Prepare balanced dataset (CSV pipeline)
```bash
python scripts/prepare_balanced_dataset.py --mode 2 --target_total_size 20000
```

---

## 🧠 Run Active Learning System

```bash
python -m methods.active_learning_in_memory
```

---

## 📊 Visualization

```bash
python scripts/plot_graph.py --input results/results1.xlsx
```

---

## 🧪 Key Engineering Goals

- Reproducible Active Learning experiments  
- Modular sampling strategy design  
- Fair comparison of strategies  
- Handling imbalanced classification properly  
- Scalable evaluation pipeline  

---
## 🔬 Research & Development Approach

This project is an **R&D-oriented machine learning system**, focused on empirical experimentation rather than production deployment.

The primary goal is to investigate and compare Active Learning strategies in terms of:

- Label efficiency over brute-force annotation  
- Empirical comparison of sampling strategies  
- Reproducible experimentation  
- Real-world noisy dataset handling  

---

## 🤝 Contributors

- Faruk Tekin  
  [@FarukTekinCENG](https://github.com/FarukTekinCENG)
- Nijat Majidli  
  [@nicat00m20](https://github.com/nicat00m20)
---

## 📄 License

MIT License

See `LICENSE` for details.
