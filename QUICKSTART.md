# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys (Optional for testing)
```bash
cp .env.example .env
# Edit .env with your NewsAPI key
```

### 3. Run a Test
```bash
# Test with sample data
python main.py --mode collect --tickers AAPL --start-date 2023-01-01 --end-date 2023-12-31
```

### 4. View Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📖 Module Overview

| Module | Purpose | Key Functions |
|--------|---------|--------------|
| `collectors.py` | Data collection | `fetch_stock_data()`, `fetch_all_news()` |
| `nlp_processor.py` | NLP pipeline | `process_news_dataframe()` |
| `temporal_features.py` | Time weighting | `calculate_temporal_weight()` |
| `numerical_features.py` | Technical indicators | `add_all_indicators()` |
| `hybrid_model.py` | Deep learning model | `build_model()`, `compile_model()` |
| `trainer.py` | Training & evaluation | `train()`, `evaluate()`, `backtest()` |

---

## 🎯 Key Concepts

### Temporal Weighting
News relevance decays exponentially:
```
weight = 0.5^(days_elapsed / 3)
```
- Day 0: weight = 1.0
- Day 3: weight = 0.5
- Day 6: weight = 0.25
- Day 30: weight ≈ 0.001

### Event Importance
Different events get different multipliers:
- **Mergers/Acquisitions**: 2.5× (highest impact)
- **Earnings Reports**: 2.0×
- **Lawsuits**: 1.8×
- **Product Launches**: 1.3×
- **Neutral News**: 1.0× (baseline)

### Model Architecture
```
Text Branch (News):
  Input(30, 768) → BiLSTM(128) × 2 → Attention → (256)

Numerical Branch (Prices):
  Input(30, 50) → GRU(64) × 2 → (64)

Fusion:
  Concat(256, 64) → Dense(128) → Dense(64) → Softmax(3)
```

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Accuracy | 67.8% |
| F1-Score | 0.66 |
| Sharpe Ratio | 1.24 |
| Max Drawdown | -12.4% |

*Note: Results vary by ticker and time period*

---

## 🔧 Troubleshooting

### Import Errors
```bash
# Make sure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Memory Issues
Reduce batch size in `configs/config.yaml`:
```yaml
training:
  batch_size: 32  # Default is 64
```

### FinBERT Download Slow
First run downloads ~400MB model. Be patient or use:
```python
# Pre-download
from transformers import AutoModel
AutoModel.from_pretrained("ProsusAI/finbert")
```

---

## 📚 Next Steps

1. **Explore Code**: Start with `main.py` and follow imports
2. **Read Paper**: `research_paper/paper.tex` explains methodology
3. **Check Configs**: `configs/config.yaml` has all hyperparameters
4. **Run Dashboard**: `streamlit run dashboard/app.py` for visualization
5. **Experiment**: Try different tickers, time periods, model configs

---

## 💡 Tips for Success

1. **Start Small**: Test with 1 ticker and 1 year of data first
2. **Cache Embeddings**: Set `cache_embeddings: true` in config
3. **Use GPU**: TensorFlow will auto-detect CUDA if available
4. **Monitor Training**: Check TensorBoard logs in `logs/`
5. **Validate Carefully**: Avoid look-ahead bias in backtesting

---

## 🎓 Learning Resources

- **FinBERT Paper**: https://arxiv.org/abs/1908.10063
- **Attention Mechanisms**: https://arxiv.org/abs/1706.03762
- **Technical Analysis**: Investopedia
- **Sharpe Ratio**: https://www.investopedia.com/terms/s/sharperatio.asp

---

## 📞 Get Help

- Check `IMPLEMENTATION_GUIDE.md` for detailed docs
- Review code comments in each module
- Open GitHub issue for bugs
- Email: your.email@university.edu

---

**Happy Predicting! 📈**
