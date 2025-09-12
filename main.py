import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from tqdm import tqdm
import numpy as np
import time

# Import your modules
from ReadTextFile import train, val, vocab_size, decode, encode
from Model import SimpleTransformerModel, ImprovedCharformerModel
from torch.cuda.amp import GradScaler, autocast


def setup_distributed():
    """Setup distributed training for Kaggle's dual T4 environment"""
    # Check if we have multiple GPUs
    if torch.cuda.device_count() <= 1:
        print("Single GPU detected, using standard training")
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0, False

    # Kaggle environment setup for distributed training
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'

    try:
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=torch.cuda.device_count(),
            rank=0
        )

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_distributed = True

        print(f"✓ Distributed training initialized on {torch.cuda.device_count()} GPUs")

    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Falling back to DataParallel")
        device = torch.device('cuda:0')
        is_distributed = False

    return device, local_rank, is_distributed


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_batch(split, batch_size, block_size, device):
    """Get a batch of training data"""
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size, block_size, device):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, min(batch_size, 16), block_size, device)  # Smaller batch for eval

            with autocast():
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


def cosine_annealing_lr(optimizer, current_step, max_steps, base_lr, min_lr=1e-6):
    """Cosine annealing learning rate schedule with warmup"""
    warmup_steps = max_steps * 0.05  # 5% warmup

    if current_step < warmup_steps:
        # Linear warmup
        lr = base_lr * current_step / warmup_steps
    else:
        # Cosine annealing
        progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def generate_sample(model, prompt="", max_tokens=200, temperature=0.8, device='cuda'):
    """Generate text sample with the model"""
    model.eval()

    if prompt:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Handle DDP wrapped models
    model_for_generation = model.module if hasattr(model, 'module') else model

    # Generate based on model type
    if hasattr(model_for_generation, 'block_size'):  # SimpleTransformerModel
        generated = model_for_generation.generate(
            context, max_tokens, temperature=temperature, top_k=50, top_p=0.95
        )
    else:  # ImprovedCharformerModel
        block_size = getattr(model_for_generation, 'effective_block_size', 512)
        generated = model_for_generation.generate(
            context, max_tokens, block_size, temperature=temperature, top_k=50, top_p=0.95
        )

    text = decode(generated[0].tolist())
    model.train()
    return text


def save_checkpoint(model, optimizer, scaler, step, loss, best_val_loss, path):
    """Save training checkpoint"""
    # Extract state dict from DDP if necessary
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'step': step,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'best_val_loss': best_val_loss,
        'vocab_size': vocab_size
    }

    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')

    # Load model state
    if hasattr(model, 'module'):  # DDP wrapped
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer and scaler if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('step', 0), checkpoint.get('best_val_loss', float('inf'))


def main():
    print("🚀 DUAL T4 CHARFORMER TRAINING SYSTEM")
    print("=" * 50)

    # Setup distributed training
    device, local_rank, is_distributed = setup_distributed()

    # Display system info
    print(f"Device: {device}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Training samples: {len(train):,}")
    print(f"Validation samples: {len(val):,}")

    # Training hyperparameters optimized for dual T4
    batch_size = 24 if torch.cuda.device_count() > 1 else 16  # Per GPU
    block_size = 512  # Context length
    learning_rate = 3e-4
    max_iters = 50000
    eval_interval = 500
    eval_iters = 50

    # Model architecture
    MODEL_DIM = 768
    depth = 8
    heads = 8
    dropout = 0.1

    accumulation_steps = 2
    gradient_clip = 1.0

    print(f"\n📊 Training Configuration:")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Effective batch size: {batch_size * max(1, torch.cuda.device_count())}")
    print(f"Block size: {block_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max iterations: {max_iters}")

    # Model selection
    print("\n🤖 Select model architecture:")
    print("1. Simple Transformer (Recommended for dual T4)")
    print("2. Improved Charformer with GBST")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")

    if choice == "1":
        print("\n✓ Using Simple Transformer Model")
        model = SimpleTransformerModel(
            num_tokens=vocab_size,
            dim=MODEL_DIM,
            depth=depth,
            heads=heads,
            block_size=block_size,
            dropout=dropout
        )
        model_path = 'simple_transformer_dual_t4.pth'
    else:
        print("\n✓ Using Improved Charformer Model")
        DOWNSAMPLE_FACTOR = 4
        gbst_blocks = ((2, 0), (3, 0), (4, 0), (5, 0))
        model = ImprovedCharformerModel(
            num_tokens=vocab_size,
            dim=MODEL_DIM,
            depth=depth,
            heads=heads,
            gbst_blocks=gbst_blocks,
            gbst_downsample_factor=DOWNSAMPLE_FACTOR
        )
        model_path = 'charformer_dual_t4.pth'

    # Move model to device
    model = model.to(device)

    # Setup distributed training or DataParallel
    if torch.cuda.device_count() > 1:
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            print("✓ Using DistributedDataParallel")
        else:
            model = torch.nn.DataParallel(model)
            print("✓ Using DataParallel")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📈 Model parameters: {total_params:,}")

    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = GradScaler()

    # Check for existing checkpoint
    start_step = 0
    best_val_loss = float('inf')

    if os.path.exists(model_path):
        print(f"\n✓ Loading checkpoint: {model_path}")
        start_step, best_val_loss = load_checkpoint(model_path, model, optimizer, scaler)
        print(f"Resuming from step {start_step}, best val loss: {best_val_loss:.4f}")

        # Generate sample
        print("\n--- Sample Generation ---")
        sample = generate_sample(model, prompt="The quick brown", max_tokens=150, device=device)
        print(sample)

        cont = input("\nContinue training? (y/n): ").strip().lower()
        if cont != 'y':
            cleanup_distributed()
            return

    print(f"\n🎯 Starting training from step {start_step}")
    print("=" * 50)

    # Training loop
    model.train()
    start_time = time.time()

    try:
        pbar = tqdm(range(start_step, max_iters), desc="Training", initial=start_step)

        for step in pbar:
            # Learning rate schedule
            current_lr = cosine_annealing_lr(optimizer, step, max_iters, learning_rate)

            # Evaluation
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model, eval_iters, batch_size, block_size, device)

                pbar.set_description(
                    f"Step {step} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | LR: {current_lr:.2e}"
                )

                # Save best model
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    best_model_path = model_path.replace('.pth', '_best.pth')
                    save_checkpoint(model, optimizer, scaler, step, losses['val'], best_val_loss, best_model_path)

                # Generate sample every 2000 steps
                if step > 0 and step % 2000 == 0:
                    print(f"\n--- Sample at step {step} ---")
                    sample = generate_sample(model, prompt="Once upon a time", max_tokens=200, device=device)
                    print(sample[:300] + "..." if len(sample) > 300 else sample)
                    print("-" * 50)

            # Training step
            xb, yb = get_batch('train', batch_size, block_size, device)

            with autocast():
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)) / accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Save checkpoint every 5000 steps
            if step > 0 and step % 5000 == 0:
                save_checkpoint(model, optimizer, scaler, step, loss.item(), best_val_loss, model_path)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    # Final save
    print(f"\n💾 Saving final model to {model_path}")
    save_checkpoint(model, optimizer, scaler, step, loss.item(), best_val_loss, model_path)

    # Load best model for generation
    best_model_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        print("🏆 Loading best model for final generation")
        load_checkpoint(best_model_path, model)

    # Final generation showcase
    print("\n" + "=" * 60)
    print("🎨 FINAL TEXT GENERATION SHOWCASE")
    print("=" * 60)

    prompts = [
        "",
        "The",
        "Once upon a time",
        "In the beginning",
        "It was a dark and stormy night",
        "The quick brown fox"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'" if prompt else f"\n[{i}/{len(prompts)}] Free generation:")
        print("-" * 40)

        generated_text = generate_sample(
            model,
            prompt=prompt,
            max_tokens=300,
            temperature=0.8,
            device=device
        )

        # Display truncated text
        display_text = generated_text[:400] + "..." if len(generated_text) > 400 else generated_text
        print(display_text)
        print("-" * 40)

    # Training summary
    total_time = time.time() - start_time
    print(f"\n📊 Training completed in {total_time / 3600:.2f} hours")
    print(f"📈 Final validation loss: {best_val_loss:.4f}")
    print(f"💾 Model saved as: {model_path}")

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()