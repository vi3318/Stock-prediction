# Windows Quick Reference - Stock Prediction System

## 🎯 Three-Step Process

### 1. Setup (One-time, ~10 minutes)

```powershell
.\WINDOWS_QUICK_SETUP.bat
```

### 2. Activate (Every session)

```powershell
.\activate_venv.bat
```

### 3. Run Experiments (5-8 hours)

```powershell
.\RUN_EXPERIMENTS.bat
```

---

## 📁 Key Files Created

| File                      | Purpose                                                                           |
| ------------------------- | --------------------------------------------------------------------------------- |
| `WINDOWS_GPU_SETUP.md`    | **Complete setup guide** with GPU config, troubleshooting, and performance tuning |
| `WINDOWS_QUICK_SETUP.bat` | **Automated installer** - Creates venv, installs all dependencies                 |
| `activate_venv.bat`       | **Environment activator** - Run before every session                              |
| `RUN_EXPERIMENTS.bat`     | **Main experiment pipeline** - Windows version of shell script                    |

---

## 🔧 Quick Commands

### Check GPU Status

```powershell
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### View Results

```powershell
type results\test_single\evaluation_metrics.json
start results\paper\figures
```

### Individual Experiments

```powershell
# Full training (2-4 hours)
python scripts\train_full_system.py --ticker AAPL --days 1000 --n-trials 100

# Quick test (15 minutes)
python scripts\train_full_system.py --ticker AAPL --days 250 --n-trials 20

# Model comparison (30-45 min)
python scripts\compare_models.py --ticker AAPL --days 1000

# Multi-sector (1-2 hours)
python examples\multi_sector_example.py

# Ablation study (30 min)
python examples\ablation_study_example.py
```

---

## ⚙️ GPU Optimization for RTX 3070

### Optimal Settings (8GB VRAM)

Edit `configs\config.yaml`:

```yaml
training:
  batch_size: 32 # Or 16 if OOM
nlp:
  batch_size: 16 # Or 8 if OOM
```

### Environment Variables

Set before training:

```powershell
$env:TF_FORCE_GPU_ALLOW_GROWTH="true"
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=512"
```

### Monitor GPU

```powershell
nvidia-smi -l 1    # Updates every second
```

---

## 🔍 Troubleshooting Quick Fixes

### Out of Memory

```yaml
# In configs/config.yaml, reduce:
training:
  batch_size: 16 # From 32
nlp:
  batch_size: 8 # From 16
```

### GPU Not Detected

```powershell
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### Slow Performance

```powershell
# High performance mode
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Check GPU usage
nvidia-smi
```

---

## 📊 Expected Results

### Training Times (RTX 3070)

- Full training: 2-4 hours
- Optuna optimization: Included in above
- Model comparison: 30-45 minutes
- Multi-sector: 1-2 hours
- Total pipeline: 5-8 hours

### GPU Utilization

- Memory: 6-7 GB / 8 GB
- Utilization: 80-95%
- Temp: 70-80°C
- Power: ~100-115W

### Accuracy Targets

- Baseline (Random Forest): 58-62%
- Basic LSTM: 60-65%
- BiLSTM + Attention: 68-72%
- **Your Hybrid Model: 80-85%** 🎯

---

## 📂 Results Location

```
results/
├── test_single/
│   └── evaluation_metrics.json     ← Your actual accuracy
├── comparison/
│   └── model_comparison.json       ← Baseline vs enhanced
├── multi_sector/
│   └── sector_performance.json     ← Multi-stock results
└── paper/
    ├── figures/                    ← 8 publication figures
    └── tables/                     ← 11 LaTeX tables
```

---

## 🎓 Documentation Hierarchy

1. **Start here**: `WINDOWS_GPU_SETUP.md` - Complete setup and troubleshooting
2. **Quick reference**: This file - Common commands
3. **Project overview**: `README.md` - System architecture
4. **Technical details**: `TECHNICAL_EXPLANATION.md`

---

## 🆘 Getting Help

1. **Check WINDOWS_GPU_SETUP.md** - Comprehensive troubleshooting section
2. **View logs**: `logs\full_training.log`
3. **Test GPU**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Verify environment**: `pip list`

---

## 💡 Pro Tips

1. **First run**? Use quick test mode:

   ```powershell
   python scripts\train_full_system.py --ticker AAPL --days 250 --n-trials 20
   ```

   Takes ~30 min instead of 2-4 hours

2. **Monitor training** in separate terminal:

   ```powershell
   nvidia-smi -l 1
   ```

3. **Save power** when not training:

   - Close browser tabs
   - Disable background apps
   - Use power saver mode when on battery

4. **Prevent sleep** during long training:

   ```powershell
   powercfg /change standby-timeout-ac 0
   ```

5. **Use cached data** after first run:
   - News and embeddings are automatically cached
   - Subsequent runs are faster

---

## ✅ Success Checklist

- [ ] Ran `WINDOWS_QUICK_SETUP.bat`
- [ ] GPU detected: `torch.cuda.is_available() == True`
- [ ] Virtual environment activated
- [ ] (Optional) NewsAPI key in `.env` file
- [ ] Sufficient disk space (10GB+)
- [ ] Stable internet connection
- [ ] Laptop plugged in for long training
- [ ] Ready to run `RUN_EXPERIMENTS.bat`

---

**Need detailed help? See [WINDOWS_GPU_SETUP.md](WINDOWS_GPU_SETUP.md)**
