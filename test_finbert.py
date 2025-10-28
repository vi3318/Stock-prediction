"""
Test script to verify FinBERT installation and loading
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

print("=" * 80)
print("FINBERT INSTALLATION TEST")
print("=" * 80)

# Check versions
print("\n1. Checking library versions:")
print(f"   - Python: {sys.version.split()[0]}")
print(f"   - PyTorch: {torch.__version__}")
try:
    import transformers
    print(f"   - Transformers: {transformers.__version__}")
except Exception as e:
    print(f"   - Transformers: ERROR - {e}")

# Check CUDA
print("\n2. Checking CUDA availability:")
print(f"   - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - CUDA version: {torch.version.cuda}")
    print(f"   - Device name: {torch.cuda.get_device_name(0)}")
    print(f"   - Device count: {torch.cuda.device_count()}")

# Test FinBERT loading
print("\n3. Testing FinBERT loading:")
try:
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ProsusAI/finbert",
        do_lower_case=True,
        trust_remote_code=True
    )
    print("   ✓ Tokenizer loaded successfully")
    print(f"     - Vocab size: {len(tokenizer)}")
    print(f"     - Max length: {tokenizer.model_max_length}")
    
    print("\n   Loading base model...")
    model = AutoModel.from_pretrained(
        "ProsusAI/finbert",
        trust_remote_code=True
    )
    print("   ✓ Base model loaded successfully")
    print(f"     - Config: {model.config.model_type}")
    print(f"     - Hidden size: {model.config.hidden_size}")
    print(f"     - Num layers: {model.config.num_hidden_layers}")
    
    print("\n   Loading sentiment model...")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=3,
        trust_remote_code=True
    )
    print("   ✓ Sentiment model loaded successfully")
    print(f"     - Num labels: {sentiment_model.config.num_labels}")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    sentiment_model.to(device)
    model.eval()
    sentiment_model.eval()
    
    print(f"\n   Models moved to device: {device}")
    
    # Test inference
    print("\n4. Testing inference:")
    test_text = "Apple reports strong quarterly earnings beating analyst expectations"
    print(f"   Test text: '{test_text}'")
    
    # Tokenize
    inputs = tokenizer(
        test_text,
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding='max_length'
    ).to(device)
    
    print("   ✓ Text tokenized successfully")
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        print(f"   ✓ Embeddings generated: shape {embeddings.shape}")
        
        # Get sentiment
        sentiment_outputs = sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(sentiment_outputs.logits, dim=-1)
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_class = torch.argmax(probs, dim=1).item()
        
        print(f"   ✓ Sentiment prediction: {sentiment_labels[predicted_class]}")
        print(f"     Probabilities:")
        for label, prob in zip(sentiment_labels, probs[0]):
            print(f"       - {label}: {prob.item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - FinBERT is working correctly!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nTroubleshooting steps:")
    print("1. Update transformers: pip install --upgrade transformers")
    print("2. Update torch: pip install --upgrade torch")
    print("3. Clear cache: rm -rf ~/.cache/huggingface/")
    print("4. Reinstall: pip uninstall transformers && pip install transformers")
    print("\n" + "=" * 80)
    import traceback
    traceback.print_exc()
    sys.exit(1)
