# Windows 11 GPU Setup Guide - Stock Prediction System

## Dell G16 RTX 3070 Configuration

This guide provides complete setup instructions for training the stock prediction model on your Windows 11 Dell G16 laptop with RTX 3070 GPU.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (5 minutes)](#quick-setup)
3. [Manual Setup (if needed)](#manual-setup)
4. [GPU Configuration](#gpu-configuration)
5. [Running Experiments](#running-experiments)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)

---

## Prerequisites

### Required Software

- **Windows 11** (your current OS ✓)
- **Python 3.8 - 3.10** (recommended: 3.10)
- **CUDA 11.8** (for PyTorch 2.0 + RTX 3070)
- **cuDNN 8.6+**
- **Git** (optional, for version control)
- **Stable internet connection** (for downloading data)

### Hardware Requirements

- **GPU**: RTX 3070 (8GB VRAM ✓)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB free space minimum
- **Expected Training Time**: 5-8 hours for full experiments

---

## Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Open PowerShell as Administrator**

   - Press `Win + X`
   - Select "Windows PowerShell (Admin)" or "Terminal (Admin)"

2. **Navigate to project directory**

   ```powershell
   cd "d:\Users\Fiona\Desktop\projects\DLNLP\Stock-prediction"
   ```

3. **Run setup script**

   ```powershell
   .\WINDOWS_QUICK_SETUP.bat
   ```

   This will automatically:

   - Check Python version
   - Create virtual environment
   - Install CUDA-enabled PyTorch
   - Install all dependencies
   - Download spaCy models
   - Verify GPU availability

4. **Activate environment and run experiments**
   ```powershell
   .\activate_venv.bat
   .\RUN_EXPERIMENTS.bat
   ```

---

## Manual Setup

### Step 1: Install Python (if not already installed)

1. Download Python 3.10 from [python.org](https://www.python.org/downloads/)
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   ```
   Should show: `Python 3.10.x`

### Step 2: Install CUDA Toolkit

Your RTX 3070 requires CUDA for GPU acceleration.

1. **Download CUDA 11.8**
   - Visit: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Select: Windows → x86_64 → 11 → exe (network)
2. **Install CUDA**

   - Run installer with default settings
   - Installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

3. **Verify CUDA installation**
   ```powershell
   nvcc --version
   ```

### Step 3: Install cuDNN

1. **Download cuDNN 8.6+ for CUDA 11.8**
   - Visit: https://developer.nvidia.com/cudnn
   - Requires free NVIDIA account
2. **Install cuDNN**
   - Extract ZIP file
   - Copy files to CUDA directory:
     ```
     cudnn\bin\*.dll     → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
     cudnn\include\*.h   → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
     cudnn\lib\*.lib     → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
     ```

### Step 4: Create Virtual Environment

1. **Navigate to project directory**

   ```powershell
   cd "d:\Users\Fiona\Desktop\projects\DLNLP\Stock-prediction"
   ```

2. **Create virtual environment**

   ```powershell
   python -m venv venv
   ```

3. **Activate virtual environment**

   ```powershell
   .\venv\Scripts\activate
   ```

   You should see `(venv)` prefix in your terminal.

### Step 5: Install PyTorch with CUDA Support

**CRITICAL**: Install PyTorch with CUDA 11.8 support FIRST, before other packages.

```powershell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU is detected:**

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:

```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3070 Laptop GPU
```

### Step 6: Install TensorFlow with GPU Support

```powershell
pip install tensorflow==2.17.0
```

**Verify TensorFlow GPU:**

```powershell
python -c "import tensorflow as tf; print('Num GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

Expected output:

```
Num GPUs: 1
```

### Step 7: Install Other Dependencies

```powershell
pip install transformers==4.30.0
pip install sentence-transformers==2.2.2
pip install yfinance==0.2.28
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install spacy nltk textblob
pip install shap lime
pip install requests beautifulsoup4 newsapi-python
pip install python-dotenv tqdm pyyaml ta
pip install optuna  # For hyperparameter optimization
```

### Step 8: Download spaCy Language Model

```powershell
python -m spacy download en_core_web_sm
```

### Step 9: Create Required Directories

```powershell
mkdir -p data\raw data\processed data\cache logs results models\saved_models visualizations
```

### Step 10: Setup News API Key (Optional but Recommended)

1. Get free API key from: https://newsapi.org/register
2. Create `.env` file in project root:
   ```
   NEWS_API_KEY=your_api_key_here
   ```

---

## GPU Configuration

### Optimize for RTX 3070 (8GB VRAM)

Your RTX 3070 has 8GB VRAM. Here are optimal settings:

#### 1. Edit `configs/config.yaml`

```yaml
training:
  batch_size: 32 # Reduce from 64 if OOM errors occur

nlp:
  batch_size: 16 # Reduce from 32 if OOM errors occur
```

#### 2. Set Environment Variables for GPU Optimization

Add to your session or create a batch file:

```powershell
# Prevent TensorFlow from allocating all GPU memory
$env:TF_FORCE_GPU_ALLOW_GROWTH="true"

# Enable mixed precision training (faster + less memory)
$env:TF_ENABLE_AUTO_MIXED_PRECISION="1"

# PyTorch memory optimization
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=512"
```

#### 3. Monitor GPU Usage

**Open a separate PowerShell window and run:**

```powershell
nvidia-smi -l 1
```

This shows real-time GPU usage, temperature, and memory consumption.

### Expected GPU Utilization

- **Memory Usage**: 6-7GB / 8GB
- **GPU Utilization**: 80-95%
- **Temperature**: 70-80°C (normal for laptop)
- **Power**: ~100-115W

---

## Running Experiments

### Complete Experiment Pipeline (5-8 hours)

1. **Activate virtual environment**

   ```powershell
   .\activate_venv.bat
   ```

2. **Run all experiments**
   ```powershell
   .\RUN_EXPERIMENTS.bat
   ```

This executes:

- **Step 1**: Full training on 1000 days AAPL (2-4 hours)

  - Downloads real stock data
  - Downloads financial news
  - Extracts FinBERT embeddings
  - Trains hybrid model
  - Optuna hyperparameter optimization (100 trials)

- **Step 2**: Baseline model comparisons (30-45 min)

  - Random Forest, Logistic Regression
  - LSTM, BiLSTM, Transformer
  - Hybrid model comparison

- **Step 3**: Multi-stock evaluation (1-2 hours)

  - 15 stocks across 5 sectors
  - Tech, Finance, Healthcare, Energy, Retail

- **Step 4**: Ablation study (30 min)

  - Component contribution analysis

- **Step 5**: Statistical significance testing (15 min)

  - t-tests, McNemar's test
  - Confidence intervals

- **Final**: Generate publication figures and tables
  - 8 figures (PNG + PDF)
  - 11 LaTeX tables

### Individual Experiment Commands

If you want to run experiments separately:

```powershell
# Training only (2-4 hours)
python scripts\train_full_system.py --ticker AAPL --days 1000 --n-trials 100

# Model comparison only (30-45 min)
python scripts\compare_models.py --ticker AAPL --days 1000

# Multi-stock evaluation (1-2 hours)
python examples\multi_sector_example.py

# Ablation study (30 min)
python examples\ablation_study_example.py

# Significance testing (15 min)
python examples\significance_testing_example.py

# Generate figures
python scripts\generate_paper_figures.py

# Generate LaTeX tables
python scripts\generate_latex_tables.py
```

---

## Troubleshooting

### Issue 1: "CUDA out of memory" Error

**Symptoms:**

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size** in `configs/config.yaml`:

   ```yaml
   training:
     batch_size: 16 # Down from 32
   nlp:
     batch_size: 8 # Down from 16
   ```

2. **Clear GPU cache** between runs:

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Enable gradient checkpointing** (trade computation for memory):
   Edit model files to use `torch.utils.checkpoint`

4. **Close other GPU applications**:
   - Close browser tabs
   - Close Discord, games, etc.
   - Check GPU usage: `nvidia-smi`

### Issue 2: "No CUDA-capable device detected"

**Symptoms:**

```
torch.cuda.is_available() returns False
```

**Solutions:**

1. **Verify NVIDIA driver is installed**:

   ```powershell
   nvidia-smi
   ```

   Should show your RTX 3070.

2. **Reinstall PyTorch with CUDA**:

   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check CUDA installation**:

   ```powershell
   nvcc --version
   ```

4. **Update NVIDIA drivers**:
   - Visit: https://www.nvidia.com/download/index.aspx
   - Download latest Game Ready Driver for RTX 3070

### Issue 3: "ImportError: DLL load failed"

**Symptoms:**

```
ImportError: DLL load failed while importing _internal
```

**Solutions:**

1. **Install Visual C++ Redistributable**:

   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart

2. **Add CUDA to PATH**:

   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
   ```

3. **Verify cuDNN installation**:
   Check files exist in CUDA directories

### Issue 4: News API Rate Limits

**Symptoms:**

```
NewsAPIException: You have made too many requests
```

**Solutions:**

1. **Get API key** from https://newsapi.org (free tier: 100 requests/day)

2. **Use cached data** (automatically enabled after first download)

3. **Reduce news collection**:
   ```yaml
   data:
     max_news_per_day: 10 # Reduce from 50
   ```

### Issue 5: Slow Training Speed

**Expected speeds on RTX 3070:**

- **Training**: ~15-20 seconds per epoch
- **With Optuna**: 2-4 hours for 100 trials
- **Total pipeline**: 5-8 hours

**If slower:**

1. **Verify GPU is being used**:

   ```powershell
   nvidia-smi
   ```

   Should show Python process using GPU

2. **Check CPU bottleneck**:

   - Task Manager → Performance
   - If CPU at 100%, reduce data loading workers

3. **Enable mixed precision**:

   ```python
   # In training scripts, add:
   torch.cuda.amp.autocast()
   ```

4. **Use SSD for data storage** (not HDD)

### Issue 6: Python Version Issues

**Symptoms:**

```
Package X requires Python >=3.10 but you have 3.8
```

**Solutions:**

1. **Verify Python version**:

   ```powershell
   python --version
   ```

2. **Create environment with specific Python**:

   ```powershell
   py -3.10 -m venv venv
   ```

3. **Update Python**: Download from python.org

### Issue 7: Insufficient Memory (RAM)

**Symptoms:**

```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce sequence length**:

   ```yaml
   model:
     sequence_length: 60 # Down from 120
   ```

2. **Reduce features**:

   ```yaml
   features:
     lookback_window: 15 # Down from 30
   ```

3. **Process data in chunks**:

   - Modify data collection to use generators

4. **Close other applications**

---

## Performance Tuning

### Maximize RTX 3070 Performance

#### 1. Power Settings

**Set Windows to High Performance mode:**

```powershell
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

**Set GPU to prefer maximum performance:**

- NVIDIA Control Panel → Manage 3D Settings → Power Management Mode → "Prefer Maximum Performance"

#### 2. Cooling Optimization

- Use on flat surface with good airflow
- Consider laptop cooling pad
- Clean air vents if dusty
- Monitor temps: `nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1`

#### 3. Batch Size Optimization

Find optimal batch size for your GPU:

```python
# Test different batch sizes
for batch_size in [8, 16, 24, 32, 40, 48]:
    try:
        # Run training
        print(f"Batch size {batch_size}: OK")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Batch size {batch_size}: OOM")
            break
```

#### 4. Mixed Precision Training

**Enables faster training with less memory:**

In `scripts/train_full_system.py`, use:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    # Training code
```

**Expected speedup**: 1.5-2x faster

#### 5. Data Loading Optimization

```python
# In DataLoader
num_workers=4,  # Adjust based on CPU cores
pin_memory=True,  # Faster CPU-GPU transfer
persistent_workers=True  # Keep workers alive
```

#### 6. Gradient Accumulation

If OOM with desired batch size:

```python
# Effective batch size = batch_size * accumulation_steps
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Results Location

After successful run, find results in:

```
results/
├── test_single/
│   ├── evaluation_metrics.json     # Your actual accuracy
│   ├── predictions.csv              # Model predictions
│   └── confusion_matrix.png         # Visual results
├── comparison/
│   ├── model_comparison.json        # Baseline vs enhanced
│   └── comparison_plots.png
├── multi_sector/
│   ├── sector_performance.json      # Results by sector
│   └── sector_analysis.png
└── paper/
    ├── figures/                     # 8 publication figures
    │   ├── figure1_architecture.pdf
    │   ├── figure2_training_curves.pdf
    │   └── ...
    └── tables/                      # 11 LaTeX tables
        ├── table1_dataset.tex
        ├── table2_hyperparameters.tex
        └── ...
```

**View your actual accuracy:**

```powershell
type results\test_single\evaluation_metrics.json
```

---

## Next Steps

After successful training:

1. **Check Results**

   ```powershell
   type results\test_single\evaluation_metrics.json
   ```

2. **View Figures**

   ```powershell
   start results\paper\figures
   ```

3. **Update Research Paper**

   - Replace placeholder results with your real metrics
   - Update `research_paper/paper.tex` with actual numbers

4. **Compile Paper**
   ```powershell
   cd research_paper
   pdflatex paper.tex
   bibtex paper
   pdflatex paper.tex
   pdflatex paper.tex
   ```

---

## Additional Resources

- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **PyTorch Windows Guide**: https://pytorch.org/get-started/locally/
- **TensorFlow GPU Setup**: https://www.tensorflow.org/install/gpu
- **RTX 3070 Specs**: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3070-3070ti/

---

## Quick Reference Commands

```powershell
# Activate environment
.\activate_venv.bat

# Run all experiments
.\RUN_EXPERIMENTS.bat

# Check GPU
nvidia-smi

# Monitor GPU in real-time
nvidia-smi -l 1

# View accuracy
type results\test_single\evaluation_metrics.json

# Individual training
python scripts\train_full_system.py --ticker AAPL --days 1000 --n-trials 100

# Quick test (faster)
python scripts\train_full_system.py --ticker AAPL --days 250 --n-trials 20
```

---

## Support

If you encounter issues:

1. Check this troubleshooting section
2. Verify GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check logs in `logs/` directory
4. Review error messages carefully

**Common issues are covered in the Troubleshooting section above.**

---

**Good luck with your training! 🚀**

Expected accuracy: **80-85%** on AAPL stock prediction with full 1000-day dataset and optimized hyperparameters.
