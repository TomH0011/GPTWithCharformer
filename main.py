import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from Config import (batch_size, block_size, device,
                    MODEL_DIM, max_iters, learning_rate,
                    DOWNSAMPLE_FACTOR, depth, heads, gbst_blocks, weight_decay,
                    eval_interval, temperature, MAX_NEW_TOKENS, accumulation_steps)
from ReadTextFile import (train, val, vocab_size, decode)
from Model import FullCharformerModel
from torch.cuda.amp import GradScaler, autocast
import time


def get_batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()  # Set the model to evaluation mode
    for split in ['train', 'val']:
        # Check only a small number of batches to get a good loss estimate quickly
        eval_iters = 20
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # We don't need autocast here as precision isn't critical for evaluation
            logits = model(X)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = Y.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set the model back to training mode
    return out


def main():
    start_time = time.perf_counter()
    # Model Initialisation
    model = FullCharformerModel(
        num_tokens=vocab_size,
        dim=MODEL_DIM,
        depth=depth,
        heads=heads,
        gbst_blocks=gbst_blocks,
        gbst_downsample_factor=DOWNSAMPLE_FACTOR
    )
    checkpoint_one = time.perf_counter()
    m = model.to(device)
    # print('Compiling model, may take a moment...')
    # m = torch.compile(m, mode="max-autotune")

    MODEL_PATH = 'charformer_model.pth'

    # Load the model if it exists, otherwise train it
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}...")
        m.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("No saved model found, starting training...")
        checkpoint_two = time.perf_counter()
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scaler = GradScaler()

        # Training Loop
        checkpoint_three = time.perf_counter()
        try:
            for steps in (pbar := tqdm(range(max_iters))):
                # Periodically evaluate the loss on train and val sets
                if steps % eval_interval == 0 or steps == max_iters - 1:
                    losses = estimate_loss(m)
                    pbar.set_description(
                        f"Step {steps}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}"
                    )

                xb, yb = get_batch('train')
                checkpoint_four = time.perf_counter()
                with autocast():
                    logits = m(xb)
                    B, T, C = logits.shape
                    logits_flat = logits.view(B * T, C)
                    targets_flat = yb.view(B * T)
                    # loss = F.cross_entropy(logits_flat, targets_flat)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)) / accumulation_steps

                scaler.scale(loss).backward()
                checkpoint_five = time.perf_counter()
                if (steps + 1) % accumulation_steps == 0:
                    checkpoint_six = time.perf_counter()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving model...")

        # Save the trained model
        print("\n--- Training Complete ---")
        print(f"Saving model to {MODEL_PATH}...")
        torch.save(m.state_dict(), MODEL_PATH)

    # --- Generate text from the trained or loaded model ---
    print("\n--- Generating New Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_output = m.generate(context, max_new_tokens=MAX_NEW_TOKENS, block_size=block_size, temperature=temperature)
    generated_text = decode(generated_output[0].tolist())
    print(generated_text)

    elapsed_time = [checkpoint_one - start_time, checkpoint_two - start_time,
                    checkpoint_three - start_time, checkpoint_four - start_time,
                    checkpoint_five - start_time, checkpoint_six - start_time]

    print("\n--- Timing Checkpoints ---")
    for i, t in enumerate(elapsed_time):
        print(f'Checkpoint {i + 1} time taken: {t:.4f} seconds')


if __name__ == '__main__':
    main()

