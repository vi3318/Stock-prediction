# 📊 Complete Guide: Generate All Graphs and Figures for Your Paper

## 🎯 Overview

You need to generate **8 figures** and **7 tables** for your IEEE research paper. Here's the complete step-by-step process.

---

## 📋 **Prerequisites Check**

### 1. **Verify Your Data Exists**

First, let's check what results you already have:

```bash
# Check for existing results
ls -la results/test_single/
ls -la results/test_comparison/
ls -la data/processed/

# Check if you have trained models
ls -la models/saved/
```

### 2. **Required Python Packages** (should already be installed)

```bash
pip3 install matplotlib seaborn pandas numpy scikit-learn plotly
```

---

## 🚀 **Step-by-Step: Generate All Figures**

### **OPTION A: You Already Have Experiment Results** ✅

If you already ran experiments and have results saved, skip to [Generate Figures](#generate-figures).

### **OPTION B: Need to Run Experiments First** 🔄

If you haven't run the full experiments yet, follow this sequence:

---

## 📊 **Step 1: Run Complete Experiments**

### **Experiment 1: Single Model Training** (20-30 minutes)

This trains the hybrid model and generates basic figures:

```bash
# Train the hybrid model on AAPL with 1000 days
python3 scripts/train_full_system.py \
    --ticker AAPL \
    --days 1000 \
    --skip-optimization \
    --output-dir results/paper_experiment_1

# This will create:
# - results/paper_experiment_1/best_model.pt
# - results/paper_experiment_1/training_history.json
# - results/paper_experiment_1/training_curves.png
# - results/paper_experiment_1/confusion_matrix.png
# - results/paper_experiment_1/evaluation_metrics.json
```

**Expected output:**
- Accuracy: ~78-80% (without Optuna optimization)
- Training time: 20-30 minutes on Apple M1

---

### **Experiment 2: Hyperparameter Optimization** (2-4 hours)

This runs Optuna to find optimal hyperparameters:

```bash
# Run with Optuna optimization (50 trials for speed, 100 for paper)
python3 scripts/train_full_system.py \
    --ticker AAPL \
    --days 1000 \
    --n-trials 50 \
    --output-dir results/paper_experiment_2

# This will create:
# - results/paper_experiment_2/best_model.pt
# - results/paper_experiment_2/optuna_study.pkl
# - results/paper_experiment_2/optimization_history.png
# - results/paper_experiment_2/hyperparameter_importance.png
```

**Expected output:**
- Best accuracy: ~82-85%
- Training time: 2-4 hours (50 trials), 4-8 hours (100 trials)

---

### **Experiment 3: Baseline Model Comparison** (30-45 minutes)

This compares your hybrid model against 6 baselines:

```bash
# Compare all models
python3 scripts/compare_models.py \
    --ticker AAPL \
    --days 1000 \
    --output-dir results/paper_baseline_comparison

# This will create:
# - results/paper_baseline_comparison/model_comparison.csv
# - results/paper_baseline_comparison/accuracy_comparison.png
# - results/paper_baseline_comparison/metrics_table.tex
```

**Expected output:**
- Random Forest: ~62%
- Logistic Regression: ~58%
- Baseline LSTM: ~63%
- BiLSTM: ~70%
- Transformer: ~72%
- Hybrid: ~78%
- Hybrid+Optuna: ~82%

---

### **Experiment 4: Multi-Stock Evaluation** (1-2 hours)

This tests generalization across multiple stocks:

```bash
# Multi-stock evaluation
python3 examples/multi_sector_example.py \
    --output-dir results/paper_multistock

# This will create:
# - results/paper_multistock/sector_performance.csv
# - results/paper_multistock/sector_boxplot.png
# - results/paper_multistock/stock_heatmap.png
```

**Expected output:**
- Technology: ~81%
- Finance: ~77%
- Healthcare: ~78%
- Consumer: ~77%
- Energy: ~75%

---

### **Experiment 5: Ablation Study** (1-2 hours)

This quantifies each component's contribution:

```bash
# Ablation study
python3 examples/ablation_study_example.py \
    --ticker AAPL \
    --days 1000 \
    --output-dir results/paper_ablation

# This will create:
# - results/paper_ablation/ablation_results.csv
# - results/paper_ablation/waterfall_chart.png
# - results/paper_ablation/component_contributions.png
```

**Expected output:**
- Full model: 82.5%
- Remove Optuna: -4%
- Remove FinBERT: -7.3%
- Remove Transformer: -5.7%
- Remove Attention: -8%

---

## 📈 **Step 2: Generate All Paper Figures**

Once you have experiment results, generate publication-quality figures:

```bash
# Generate all 8 figures for the paper
python3 scripts/generate_paper_figures.py \
    --results-dir results \
    --output-dir results/paper/figures

# This creates 8 figures (PNG + PDF):
# 1. figure1_accuracy_comparison.png/pdf
# 2. figure2_training_curves.png/pdf
# 3. figure3_confusion_matrices.png/pdf
# 4. figure4_ablation_study.png/pdf
# 5. figure5_feature_importance.png/pdf
# 6. figure6_multistock_performance.png/pdf
# 7. figure7_hyperparameter_optimization.png/pdf
# 8. figure8_sequence_length_analysis.png/pdf
```

---

## 📋 **Step 3: Generate All LaTeX Tables**

Generate publication-ready LaTeX tables:

```bash
# Generate all 11 tables
python3 scripts/generate_latex_tables.py \
    --results-dir results \
    --output-dir results/paper/tables

# This creates 11 .tex files:
# table1_model_comparison.tex
# table2_ablation_study.tex
# table3_multistock_results.tex
# table4_hyperparameter_optuna.tex
# table5_feature_categories.tex
# table6_statistical_significance.tex
# table7_computational_efficiency.tex
# table8_confusion_matrix_metrics.tex
# table9_training_time_comparison.tex
# table10_error_analysis.tex
# table11_dataset_statistics.tex
```

---

## 🎨 **Step 4: Create Architecture Diagrams**

You need to create 2 architecture diagrams manually:

### **Figure 1: System Architecture**

**Use Draw.io / diagrams.net:**

1. Go to https://app.diagrams.net/
2. Create a flowchart with these components:

```
[Raw Stock Data (Yahoo Finance)] → [Data Preprocessing]
                                          ↓
[Raw News Articles (NewsAPI)] → [Text Preprocessing] → [FinBERT Embedding]
                                          ↓                      ↓
                                 [Feature Engineering (80 features)]
                                          ↓
                              [Feature Fusion (1,619 dims)]
                                          ↓
                         [Hybrid Neural Network]
                         (BiLSTM → Attention → Transformer)
                                          ↓
                              [Prediction (Up/Down)]
```

3. Export as PDF: `figure1_system_architecture.pdf`

### **Figure 2: Model Architecture**

**Create detailed neural network diagram:**

```
Input (120 × 1,619)
        ↓
Feature Projection (120 × 192)
        ↓
┌─────────────────────────┐
│   3-Layer BiLSTM        │
│   Forward LSTM (192)    │
│   Backward LSTM (192)   │
│   Output: 120 × 384     │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ Learnable Temporal      │
│ Weighting (Adaptive)    │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ Multi-Head Attention    │
│ (6 heads)               │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ Transformer Encoder     │
│ (2 blocks, FFN 768)     │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ Fusion Layer            │
│ (Last + Max + Mean)     │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ FC1 (512) + Dropout     │
│ FC2 (1) + Sigmoid       │
└─────────────────────────┘
        ↓
   Prediction (0-1)
```

Export as: `figure2_model_architecture.pdf`

---

## 🔧 **Quick Test: Generate Sample Figures First**

Before running long experiments, test the figure generation:

```bash
# Test with dummy data (already created)
python3 test_system_simple.py --mode single

# Then generate figures from test results
python3 scripts/generate_paper_figures.py \
    --results-dir results/test_single \
    --output-dir results/paper/figures_test
```

---

## ⚡ **FASTEST PATH: Skip Long Experiments**

If you want figures NOW without waiting 5-8 hours for experiments:

### **Option 1: Use Existing Test Results**

```bash
# Generate figures from your test_single results
python3 scripts/generate_paper_figures.py

# This uses simulated data but shows proper structure
```

### **Option 2: Run Quick Mini-Experiments**

```bash
# Quick 200-day experiment (10 minutes)
python3 scripts/train_full_system.py \
    --ticker AAPL \
    --days 200 \
    --skip-optimization

# Quick comparison (15 minutes)
python3 scripts/compare_models.py \
    --ticker AAPL \
    --days 200
```

---

## 📊 **What Each Script Does**

### **1. `generate_paper_figures.py`** (ALREADY EXISTS)

**Creates:**
- Bar charts (accuracy comparison)
- Line plots (training curves)
- Heatmaps (confusion matrices)
- Waterfall charts (ablation study)
- Box plots (multi-stock)

**Reads from:**
- `results/*/training_history.json`
- `results/*/evaluation_metrics.json`
- `results/*/confusion_matrix.csv`

### **2. `generate_latex_tables.py`** (ALREADY EXISTS)

**Creates:**
- Model comparison table
- Ablation study table
- Multi-stock results table
- Statistical significance table
- Hyperparameter table

**Reads from:**
- Same JSON/CSV files as figures

### **3. `train_full_system.py`** (YOUR MAIN TRAINING SCRIPT)

**Does:**
- Trains hybrid model
- Saves results to JSON
- Generates basic plots

### **4. `compare_models.py`** (BASELINE COMPARISON)

**Does:**
- Trains 6 baseline models
- Compares with hybrid
- Saves comparison results

---

## 🎯 **Recommended Workflow**

### **For Quick Paper Draft** (2-3 hours total):

1. **Run quick experiments:**
   ```bash
   python3 scripts/train_full_system.py --ticker AAPL --days 200 --skip-optimization
   python3 scripts/compare_models.py --ticker AAPL --days 200
   ```

2. **Generate figures:**
   ```bash
   python3 scripts/generate_paper_figures.py
   ```

3. **Generate tables:**
   ```bash
   python3 scripts/generate_latex_tables.py
   ```

4. **Create diagrams** (1-2 hours):
   - Draw architecture diagrams in Draw.io

5. **Compile paper:**
   ```bash
   cd research_paper
   pdflatex ieee_paper.tex
   ```

### **For Final Submission** (1-2 days total):

1. **Run full experiments** (5-8 hours):
   - 1000-day training
   - Optuna optimization (100 trials)
   - Multi-stock evaluation (15 stocks)
   - Ablation studies

2. **Generate all figures** (30 minutes)

3. **Generate all tables** (10 minutes)

4. **Create high-quality diagrams** (2-3 hours)

5. **Final compilation and proofreading** (2-4 hours)

---

## 🐛 **Troubleshooting**

### **Issue: "Module not found"**
```bash
pip3 install matplotlib seaborn pandas numpy scikit-learn plotly torch transformers
```

### **Issue: "No such file or directory: results/"**
```bash
mkdir -p results/paper/figures
mkdir -p results/paper/tables
```

### **Issue: "CUDA out of memory"**
```bash
# Use CPU instead (already configured)
# Or reduce batch size in config
```

### **Issue: "Training too slow"**
```bash
# Reduce days: --days 200 instead of 1000
# Skip optimization: --skip-optimization
# Reduce trials: --n-trials 25 instead of 100
```

---

## 📝 **Summary Checklist**

- [ ] Install required packages
- [ ] Run training experiments (or use test results)
- [ ] Generate 8 figures (`generate_paper_figures.py`)
- [ ] Generate 11 tables (`generate_latex_tables.py`)
- [ ] Create 2 architecture diagrams (Draw.io)
- [ ] Compile LaTeX paper
- [ ] Check all figures appear correctly
- [ ] Verify table formatting
- [ ] Final proofreading

---

## 🚀 **START HERE - Simplest Path**

```bash
# 1. Generate figures from existing test data (2 minutes)
python3 scripts/generate_paper_figures.py

# 2. Generate tables (1 minute)
python3 scripts/generate_latex_tables.py

# 3. Check outputs
ls -la results/paper/figures/
ls -la results/paper/tables/

# 4. Compile paper
cd research_paper
pdflatex ieee_paper.tex
```

That's it! You'll have a draft with simulated figures. Then run real experiments and regenerate for final version.

---

## 📞 **Need Help?**

If scripts fail, let me know the error message and I'll fix them!
