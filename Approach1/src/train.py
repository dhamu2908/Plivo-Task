import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU (reduced for 4GB GPU)")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate gradients to simulate larger batch")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=128, help="Reduced from 256 for memory efficiency")
    ap.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision training")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    
    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.to(args.device)
    model.train()

    # Use AdamW with memory-efficient settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    
    # Calculate total steps with gradient accumulation
    total_steps = (len(train_dl) // args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Initialize mixed precision scaler if available and enabled
    scaler = None
    if args.fp16 and AMP_AVAILABLE and args.device == "cuda":
        scaler = GradScaler()
        print("Mixed precision training (FP16) enabled")

    print(f"Training configuration:")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Batch size per step: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Total training steps: {total_steps}")
    
    # Initialize gradients
    optimizer.zero_grad()
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            # Mixed precision forward pass
            if scaler is not None:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

            running_loss += loss.item() * args.gradient_accumulation_steps

            # Update weights after accumulating gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
