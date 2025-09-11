import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from Config import (batch_size, block_size, device,
                    MODEL_DIM, max_iters, learning_rate,
                    DOWNSAMPLE_FACTOR, depth, heads, gbst_blocks, weight_decay,
                    eval_interval, temperature, MAX_NEW_TOKENS, accumulation_steps)
from ReadTextFile import (train, val, vocab_size, decode, encode)
from Model import ImprovedCharformerModel, SimpleTransformerModel
from torch.cuda.amp import GradScaler, autocast
import time
import numpy as np


def get_batch(split, batch_size_override=None):
    """Get a batch of data with optional batch size override"""
    data = train if split == 'train' else val
    bs = batch_size_override if batch_size_override else batch_size
    ix = torch.randint(len(data) - block_size, (bs,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size_override=min(batch_size, 32))  # Use smaller batch for eval
            with autocast():
                logits = model(X)
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                targets_flat = Y.view(B * T)
                loss = F.cross_entropy(logits_flat, targets_flat)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def cosine_annealing_lr(optimizer, current_step, max_steps, min_lr=1e-6):
    """Cosine annealing learning rate schedule"""
    if current_step < max_steps * 0.1:  # Warmup for first 10%
        lr = learning_rate * current_step / (max_steps * 0.1)
    else:
        progress = (current_step - max_steps * 0.1) / (max_steps * 0.9)
        lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def generate_sample(model, prompt="", max_tokens=200, temperature=0.8):
    """Generate a sample of text with optional prompt"""
    model.eval()

    if prompt:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Check which model type we're using
    if hasattr(model, 'block_size'):  # SimpleTransformerModel
        generated = model.generate(context, max_tokens, temperature=temperature, top_k=50, top_p=0.95)
    else:  # ImprovedCharformerModel
        generated = model.generate(context, max_tokens, block_size, temperature=temperature, top_k=50, top_p=0.95)

    text = decode(generated[0].tolist())
    model.train()
    return text


def main():
    print("CHARFORMER TRAINING SYSTEM")

    # Model selection
    print("\nSelect model architecture:")
    print("1. Improved Charformer with GBST (recommended for subword learning)")
    print("2. Simple Transformer (recommended for character-level)")

    model_choice = input("Enter choice (1 or 2): ").strip()

    if model_choice == "1":
        print("\n✓ Using Improved Charformer Model")
        model = ImprovedCharformerModel(
            num_tokens=vocab_size,
            dim=MODEL_DIM,
            depth=depth,
            heads=heads,
            gbst_blocks=gbst_blocks,
            gbst_downsample_factor=DOWNSAMPLE_FACTOR
        )
        MODEL_PATH = 'improved_charformer_model.pth'
    else:
        print("\n✓ Using Simple Transformer Model")
        model = SimpleTransformerModel(
            num_tokens=vocab_size,
            dim=MODEL_DIM,
            depth=depth,
            heads=heads,
            block_size=block_size,
            dropout=0.2
        )
        MODEL_PATH = 'simple_transformer_model.pth'

    m = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in m.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Training Samples: {len(train):,}")
    print(f"Validation Samples: {len(val):,}")

    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"\n✓ Loading saved model from {MODEL_PATH}")
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        # Generate sample
        print("\n--- Sample Generation ---")
        sample = generate_sample(m, prompt="The ", max_tokens=200)
        print(sample)

        continue_training = input("\nContinue training? (y/n): ").strip().lower()
        if continue_training != 'y':
            return
    else:
        print("\n✗ No saved model found, starting fresh training...")

    # Training setup
    optimiser = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()

    # Training metrics
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10  # Early stopping patience

    print(f"\n--- Training Configuration ---")
    print(f"Max Iterations: {max_iters}")
    print(f"Batch Size: {batch_size}")
    print(f"Block Size: {block_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Accumulation Steps: {accumulation_steps}")

    print("\n--- Starting Training ---")

    try:
        for steps in (pbar := tqdm(range(max_iters), desc="Training")):
            # Adjust learning rate
            current_lr = cosine_annealing_lr(optimiser, steps, max_iters)

            # Evaluation
            if steps % eval_interval == 0 or steps == max_iters - 1:
                losses = estimate_loss(m)
                pbar.set_description(
                    f"Step {steps}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, LR {current_lr:.2e}"
                )

                # Early stopping check
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    patience_counter = 0
                    # Save best model
                    torch.save(m.state_dict(), MODEL_PATH.replace('.pth', '_best.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"\nEarly stopping triggered after {steps} steps")
                        break

                # Generate sample every 5000 steps
                if steps > 0 and steps % 5000 == 0:
                    print(f"\n--- Sample at step {steps} ---")
                    sample = generate_sample(m, prompt="Once upon a time", max_tokens=100, temperature=0.8)
                    print(sample[:200] + "..." if len(sample) > 200 else sample)
                    print("-" * 50)

            # Training step
            xb, yb = get_batch('train')

            # Use autocast for efficiency
            with autocast():
                logits = m(xb)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)) / accumulation_steps

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (steps + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

            # Checkpoint saving
            if steps > 0 and steps % 10000 == 0:
                checkpoint_path = f"checkpoint_{steps}.pth"
                torch.save({
                    'step': steps,
                    'model_state_dict': m.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': loss.item(),
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                print(f"\n✓ Checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")

    # Save final model
    print(f"\n✓ Saving final model to {MODEL_PATH}")
    torch.save(m.state_dict(), MODEL_PATH)

    # Load best model if it exists
    best_model_path = MODEL_PATH.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        print(f"✓ Loading best model from validation")
        m.load_state_dict(torch.load(best_model_path, map_location=device))

    # Final text generation
    print("\n" + "=" * 50)
    print("FINAL TEXT GENERATION")
    print("=" * 50)

    prompts = [
        "",  # No prompt
        "The ",
        "Once upon a time",
        "In the beginning",
        "It was a dark and stormy night",
        "According to all known laws of aviation"
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'" if prompt else "\nNo prompt (free generation):")
        generated_text = generate_sample(m, prompt=prompt, max_tokens=MAX_NEW_TOKENS, temperature=temperature)
        print("-" * 30)
        print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
        print("-" * 30)


if __name__ == '__main__':
    main()