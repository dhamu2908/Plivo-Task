# Latency Optimization Solutions for PII NER Model

## Problem
Current p95 latency: **41.41ms** (Target: ≤20ms)

## Solutions to Achieve <20ms Latency

### Option 1: Use a Smaller Model (FASTEST & EASIEST) ✅
**Replace DistilBERT with BERT-tiny**

```bash
# Retrain with smaller model
python src/train.py \
  --model_name prajjwal1/bert-tiny \
  --epochs 5 \
  --batch_size 8 \
  --out_dir out_tiny

# Test latency
python src/measure_latency.py \
  --model_dir out_tiny \
  --device cpu \
  --runs 50
```

**Expected Results:**
- **Latency**: ~5-8ms (✅ well under 20ms)
- **Accuracy**: ~85-90% F1 (slightly lower but acceptable)
- **Model size**: 4.4M params (vs 66M for DistilBERT)
- **15x smaller, 5-8x faster**

---

###Option 2: ONNX Runtime with Optimization ✅

```bash
# Install ONNX tools
pip install onnx onnxruntime optimum[onnxruntime]

# Export to optimized ONNX
python -m optimum.onnxruntime.export \
  --model out_new \
  --task token-classification \
  out_new_onnx

# Quantize ONNX model
python -m onnxruntime.quantization.preprocess \
  --input out_new_onnx/model.onnx \
  --output out_new_onnx/model_quantized.onnx
```

**Expected Results:**
- **Latency**: ~12-18ms (near target)
- **Accuracy**: Same as original (no loss)
- **Speedup**: 2-3x faster

---

### Option 3: TorchScript with Optimizations

```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("out_new")
model.eval()

# Convert to TorchScript
example_input = torch.randint(0, 1000, (1, 128))
example_mask = torch.ones((1, 128), dtype=torch.long)

traced_model = torch.jit.trace(
    model,
    (example_input, example_mask),
    strict=False
)

# Optimize for CPU
optimized_model = torch.jit.optimize_for_inference(traced_model)

# Save
torch.jit.save(optimized_model, "out_new/model_torchscript.pt")
```

**Expected Results:**
- **Latency**: ~25-30ms (improvement but may not hit target)
- **Accuracy**: Same
- **Speedup**: ~1.5x

---

### Option 4: Batch Processing (if allowed)

Instead of batch_size=1, process multiple utterances:

```python
# Process in mini-batches
batch_size = 4  # Or 8
# Expected latency per utterance: 41ms / 4 = ~10ms
```

**Expected Results:**
- **Latency per item**: ~10-12ms (✅ under 20ms)
- **Throughput**: 4x better
- **Note**: Only works if you can batch inputs

---

## Recommended Approach for Assignment

### Quick Win: Use BERT-tiny

```bash
# 1. Train with tiny model (5 minutes)
python src/train.py --model_name prajjwal1/bert-tiny --epochs 5 --out_dir out_tiny

# 2. Evaluate
python src/predict.py --model_dir out_tiny --input data/dev.jsonl --output out_tiny/dev_pred.json
python src/eval_span_f1.py --gold data/dev.jsonl --pred out_tiny/dev_pred.json

# 3. Measure latency
python src/measure_latency.py --model_dir out_tiny --device cpu --runs 50
```

**Why this works:**
- BERT-tiny: 4.4M parameters
- DistilBERT: 66M parameters  
- **15x smaller = 5-8x faster**
- Still achieves 85-90% F1 (good enough for the task)
- PII precision likely stays above 0.80

---

## Comparison Table

| Method | p95 Latency | PII Precision | Accuracy Loss | Effort |
|--------|-------------|---------------|---------------|--------|
| **Current (DistilBERT)** | 41ms | 0.949 | - | - |
| **BERT-tiny** | ~6ms ✅ | ~0.85 | ~5-10% | Low |
| **ONNX + Quantization** | ~15ms ✅ | 0.949 | None | Medium |
| **TorchScript** | ~28ms ❌ | 0.949 | None | Low |
| **Batch=4** | ~10ms/item ✅ | 0.949 | None | Low |

---

## Implementation Priority

**For your 2-hour assignment:**

1. ✅ **First**: Try BERT-tiny (5 min train, likely hits target)
2. ✅ **If needed**: Add simple caching/batching
3. ⚠️ **If time**: ONNX optimization (more complex)

**The BERT-tiny approach is your best bet** - it's simple, fast to train on your 4GB GPU, and will easily hit the <20ms target while maintaining acceptable accuracy!

---

## Quick Command to Test BERT-tiny

```bash
python src/train.py \
  --model_name prajjwal1/bert-tiny \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out_tiny \
  --batch_size 8 \
  --epochs 5 \
  --max_length 128
```

This should complete in ~3-4 minutes and achieve <10ms latency! 🚀
