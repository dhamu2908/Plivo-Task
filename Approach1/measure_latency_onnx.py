"""
Ultra-fast inference using ONNX Runtime with quantization.
This provides the best CPU inference performance.
"""
import json
import time
import argparse
import statistics
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠ ONNX Runtime not available. Install with: pip install onnxruntime")

from transformers import AutoTokenizer


def main():
    if not ONNX_AVAILABLE:
        print("Please install onnxruntime: pip install onnxruntime")
        return
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default="out_new/model.onnx")
    parser.add_argument("--model_dir", default="out_new")
    parser.add_argument("--input", default="data/dev.jsonl")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    print("Loading ONNX model...")
    session = ort.InferenceSession(
        args.onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        print("No texts found in input file.")
        return

    times_ms = []

    # Warmup
    print("Warming up...")
    for _ in range(5):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
            padding="max_length"
        )
        _ = session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)
            }
        )

    print(f"Running {args.runs} inference runs...")
    for i in range(args.runs):
        t = texts[i % len(texts)]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
            padding="max_length"
        )
        
        start = time.perf_counter()
        _ = session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)
            }
        )
        end = time.perf_counter()
        
        times_ms.append((end - start) * 1000)

    p50 = statistics.median(times_ms)
    p95 = statistics.quantiles(times_ms, n=20)[18]

    print(f"\nONNX Runtime Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")
    
    # Compare with original
    original_p95 = 41.41
    speedup = original_p95 / p95
    print(f"\nSpeedup vs original PyTorch FP32:")
    print(f"  Original p95: {original_p95:.2f} ms")
    print(f"  ONNX p95: {p95:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x faster")
    
    if p95 <= 20:
        print(f"  ✓✓✓ Target achieved! (p95 ≤ 20ms)")
    else:
        print(f"  ⚠ Still {p95-20:.2f}ms above 20ms target")


if __name__ == "__main__":
    main()
