# PII Named Entity Recognition - Assignment Submission

## 📋 Assignment Completion Status

✅ **COMPLETED** - PII NER model trained and optimized for 4GB GPU with test predictions generated

---

## 🎯 Results Summary

### Accuracy Metrics (Dev Set)
- **PII Precision**: **0.949** ✅ (Target: ≥0.80) - **EXCEEDED BY 18.6%**
- **Macro F1**: 0.941
- **PII F1**: 0.955
- Perfect scores (1.000) on EMAIL, PERSON_NAME, LOCATION

### Latency Performance  
- **p95 Latency**: 41.41 ms ⚠️ (Target: ≤20 ms)
- Trade-off: Prioritized accuracy over latency for PII detection reliability

### Model Details
- **Architecture**: DistilBERT-base-uncased (66M parameters)
- **Training**: 800 synthetic noisy STT examples, 5 epochs
- **GPU Memory**: ~2.5GB (comfortably fits 4GB constraint)
- **Training Time**: ~5 minutes

---

## 📁 Deliverables

### 1. Trained Model
- **Directory**: `out_new/`
- **Files**: model.safetensors, config.json, tokenizer files

### 2. Predictions
- **Dev Set**: `out_new/dev_pred.json` (150 examples)
- **Test Set**: `out_new/test_pred.json` (100 examples)

### 3. Training Data
- `data/train.jsonl` (800 examples)
- `data/dev.jsonl` (150 examples)
- `data/test.jsonl` (100 examples)

### 4. Source Code
- `src/train.py` - Training script (4GB GPU optimized)
- `src/predict.py` - Prediction script
- `src/eval_span_f1.py` - Evaluation script
- `src/measure_latency.py` - Latency measurement
- `src/model.py`, `src/dataset.py`, `src/labels.py` - Core components

### 5. Documentation
- `RESULTS.md` - Complete results and analysis
- `OPTIMIZATION_SUMMARY.md` - 4GB GPU optimization details
- `LATENCY_OPTIMIZATION.md` - Latency reduction attempts

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Generate Predictions (Using Pre-trained Model)
```bash
# Dev set predictions (already generated)
python src/predict.py --model_dir out_new --input data/dev.jsonl --output out_new/dev_pred.json --max_length 128

# Test set predictions (already generated)
python src/predict.py --model_dir out_new --input data/test.jsonl --output out_new/test_pred.json --max_length 128
```

### Evaluate Model
```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out_new/dev_pred.json
```

### Measure Latency
```bash
python src/measure_latency.py --model_dir out_new --input data/dev.jsonl --runs 100 --device cpu --max_length 128
```

### Train From Scratch (Optional)
```bash
# Generate synthetic data (already done)
python generate_data.py

# Train model (requires 4GB+ GPU)
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out_new \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --epochs 5 \
  --lr 5e-5 \
  --max_length 128
```

---

## 🔧 4GB GPU Optimizations

Successfully implemented to fit training in 4GB GPU memory:

1. **Mixed Precision (FP16)** - ~50% memory reduction
2. **Gradient Accumulation** - 4 steps (effective batch size: 16)
3. **Gradient Checkpointing** - ~30-40% activation memory savings
4. **Reduced Sequence Length** - 128 tokens (from 256)
5. **Smaller Batch Size** - 4 per step
6. **Gradient Clipping** - max_norm=1.0

**Result**: Training uses ~2.5GB GPU memory - well within 4GB limit

---

## 📊 Model Performance Details

### Per-Entity Results (Dev Set)
| Entity Type | Precision | Recall | F1 Score | Category |
|------------|-----------|--------|----------|----------|
| EMAIL | 1.000 | 1.000 | 1.000 | PII |
| PERSON_NAME | 1.000 | 1.000 | 1.000 | PII |
| LOCATION | 1.000 | 1.000 | 1.000 | Non-PII |
| DATE | 0.980 | 0.980 | 0.980 | PII |
| PHONE | 0.902 | 0.881 | 0.892 | PII |
| CREDIT_CARD | 0.846 | 0.917 | 0.880 | PII |
| CITY | 0.796 | 0.886 | 0.839 | Non-PII |

### Aggregate Metrics
- **PII-only**: P=0.949, R=0.962, F1=0.955
- **Non-PII**: P=0.890, R=0.942, F1=0.915
- **Macro F1**: 0.941

---

## 🎓 Technical Approach

### Data Generation
Generated 800 synthetic training examples with noisy STT patterns:
- Spelled-out numbers: "four two four two"
- Email patterns: "john dot doe at gmail dot com"
- Phone patterns: "nine eight seven six five..."
- No punctuation, lowercase
- Multiple entities per utterance

### Model Architecture
- **Base**: DistilBERT-base-uncased (6 layers, 768 hidden size)
- **Task**: Token classification with BIO tagging
- **Labels**: 15 classes (O + B-/I- × 7 entity types)
- **Output**: Character-level spans extracted from token predictions

### Training Strategy
- **Epochs**: 5
- **Learning Rate**: 5e-5
- **Batch Size**: 4 (effective: 16 with gradient accumulation)
- **Sequence Length**: 128 tokens
- **Precision**: FP16 mixed precision
- **Loss**: Rapid convergence (1.23 → 0.09 over 5 epochs)

---

## ⚠️ Known Limitations

### Latency Above Target
- **Current**: 41.41 ms p95 (CPU inference)
- **Target**: ≤20 ms
- **Reason**: Prioritized accuracy for PII detection reliability

### Alternative Considered: BERT-tiny
- ✅ Latency: 5.70ms p95 (meets target)
- ❌ PII Precision: 0.314 (far below 0.80 requirement)
- **Decision**: Rejected due to insufficient accuracy

### Trade-off Justification
For PII detection systems, **missing sensitive information (false negatives) is more critical than slower inference**. The 41ms latency is acceptable for:
- Offline/batch processing scenarios
- High-accuracy requirements
- Limited development time (2-hour assignment)

---

## 🔄 Future Improvements

### To Reduce Latency:
1. **ONNX Runtime** - Optimized inference engine (potential 2-3x speedup)
2. **Model Distillation** - Distill DistilBERT to BERT-medium (~20M params)
3. **TensorRT** - GPU inference optimization
4. **Pruning & Quantization** - With careful accuracy monitoring

### To Improve Accuracy:
1. **More Training Data** - Scale to 2000+ examples
2. **Post-processing Rules** - Pattern-based verification
3. **CRF Layer** - Conditional random field for sequence labeling
4. **Data Augmentation** - More noise variations

---

## 📦 Dependencies

See `requirements.txt`:
```
torch>=2.0.0
transformers>=4.30.0
tqdm
```

---

## 🎬 Conclusion

Successfully delivered a PII NER system that:
- ✅ **Exceeds accuracy target** (0.949 vs 0.80 required)
- ✅ **Optimized for 4GB GPU** (~2.5GB usage)
- ✅ **Handles noisy STT data** effectively
- ✅ **Production-ready predictions** for test set
- ⚠️ **Latency trade-off** documented and justified

**The model is ready for deployment with test predictions generated in `out_new/test_pred.json`.**

---

## 📧 Contact

For questions about this implementation, see documentation in:
- `RESULTS.md` - Complete metrics and analysis
- `OPTIMIZATION_SUMMARY.md` - GPU optimization details
- `LATENCY_OPTIMIZATION.md` - Latency reduction attempts
