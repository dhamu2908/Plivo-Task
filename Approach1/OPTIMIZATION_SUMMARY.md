# 4GB GPU Optimization Summary

## Task Overview
PII Entity Recognition for Noisy STT Transcripts
- Detect: CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE, CITY, LOCATION
- PII entities (high precision): CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE
- Non-PII: CITY, LOCATION
- Target: p95 latency ≤ 20ms on CPU, PII precision ≥ 0.80

## Memory Optimizations Applied for 4GB GPU

### 1. **Mixed Precision Training (FP16)**
- Reduces memory usage by ~50%
- Automatic loss scaling to prevent underflow
- Enabled by default in training

### 2. **Gradient Accumulation**
- Effective batch size: 16 (4 × 4 accumulation steps)
- Actual batch size per GPU step: 4
- Simulates larger batch training without memory overhead

### 3. **Gradient Checkpointing**
- Trades computation for memory
- Recomputes activations during backward pass
- Saves ~30-40% memory for transformer models
- Enabled automatically in train.py

### 4. **Reduced Sequence Length**
- Max length: 128 tokens (reduced from 256)
- Sufficient for most STT utterances
- Reduces memory quadratically for attention

### 5. **Gradient Clipping**
- Max norm: 1.0
- Prevents exploding gradients
- Improves training stability

### 6. **Memory-Efficient Optimizer Settings**
- AdamW with eps=1e-8
- Proper warmup: 10% of total steps
- Linear schedule with warmup

## Training Configuration

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out_new \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --epochs 3 \
  --lr 5e-5 \
  --max_length 128 \
  --fp16 \
  --gradient_checkpointing
```

## Dataset Generation

Generated synthetic noisy STT data:
- **Training**: 800 examples
- **Dev**: 150 examples  
- **Test**: 100 examples (unlabeled)

Characteristics:
- Spelled-out numbers: "four two four two"
- Email patterns: "name dot surname at gmail dot com"
- Phone patterns: "nine eight seven six five four three two one zero"
- No punctuation, lowercase, realistic STT noise

## Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
  - 66M parameters
  - 6 layers, 768 hidden size
  - Fast inference (~2-5ms per example on CPU)
- **Task Head**: Token classification for 15 BIO tags
- **Label Set**: O, B-ENTITY, I-ENTITY for 7 entity types

## Memory Usage Estimate

With current optimizations on 4GB GPU:
- Model: ~270MB
- Optimizer states: ~540MB  
- Gradients: ~270MB (with checkpointing)
- Activations: ~500-800MB (with FP16 + checkpointing)
- **Total**: ~2-2.5GB (comfortably fits in 4GB)

## Performance Expectations

### Training Speed
- ~2 iterations/second on 4GB GPU
- ~100 seconds per epoch (200 batches)
- Total training time: ~5 minutes for 3 epochs

### Inference Speed
- DistilBERT on CPU: 2-10ms per example
- Well within p95 ≤ 20ms target

### Model Quality
- DistilBERT for NER: F1 typically 0.85-0.92
- PII precision target ≥ 0.80: achievable
- May need post-processing rules for high precision

## Evaluation Pipeline

```bash
# 1. Generate predictions
python src/predict.py \
  --model_dir out_new \
  --input data/dev.jsonl \
  --output out_new/dev_pred.json \
  --max_length 128

# 2. Evaluate span F1 and PII metrics
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out_new/dev_pred.json

# 3. Measure CPU latency
python src/measure_latency.py \
  --model_dir out_new \
  --input data/dev.jsonl \
  --runs 50 \
  --device cpu \
  --max_length 128
```

## Key Files Modified

1. **src/train.py**
   - Added FP16 mixed precision training
   - Implemented gradient accumulation
   - Added gradient checkpointing support
   - Reduced default batch_size and max_length
   - Added gradient clipping

2. **src/predict.py**
   - Updated max_length default to 128
   - Optimized inference with proper torch.no_grad()

3. **src/measure_latency.py**
   - Updated max_length default to 128
   - Set default device to CPU for latency measurement

4. **generate_data.py** (NEW)
   - Synthetic noisy STT data generator
   - Realistic speech patterns
   - Multiple entity types per utterance

## Further Optimizations (if needed)

If you encounter OOM errors:
1. Reduce `--batch_size` to 2
2. Increase `--gradient_accumulation_steps` to 8
3. Reduce `--max_length` to 96
4. Use smaller model: "prajjwal1/bert-tiny" (4.4M params)

## Notes

- All optimizations are transparent to model quality
- FP16 maintains same accuracy as FP32
- Gradient accumulation produces identical results to larger batches
- Training is fully resumable from checkpoints
