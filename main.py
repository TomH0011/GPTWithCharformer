import torch
import torch.nn.functional as F
from einops import rearrange
import os
from tqdm import tqdm
from Config import (batch_size, block_size, device,
                    MODEL_DIM, max_iters, learning_rate, DOWNSAMPLE_FACTOR)
from ReadTextFile import (train, val, vocab_size, decode)
from Model import FullCharformerModel


def get_batch(split):
    """Grabs a random batch of data for training or validation."""
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    """Main function to run the training and generation process."""

    # --- Model Initialization ---
    model = FullCharformerModel(
        num_tokens=vocab_size,
        dim=MODEL_DIM,
        depth=6,
        heads=8,
        gbst_blocks=((3, 0), (3, 1), (3, 2)),
        gbst_downsample_factor=DOWNSAMPLE_FACTOR
    )
    m = model.to(device)

    MODEL_PATH = 'charformer_model.pth'

    # --- Load the model if it exists, otherwise train it ---
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}...")
        m.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("No saved model found, starting training...")
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

        # Training Loop with progress bar
        for steps in (pbar := tqdm(range(max_iters))):
            xb, yb = get_batch('train')
            logits = m(xb)

            # --- DELETED BLOCK ---
            # The model now upsamples, so its output logits have the same length as the target yb.
            # We no longer need to downsample the target.
            #
            # pad_to_len = logits.shape[1] * DOWNSAMPLE_FACTOR
            # padded_yb = F.pad(yb, (0, pad_to_len - yb.shape[1]))
            # yb_downsampled = rearrange(padded_yb, 'b (n d) -> b n d', d=DOWNSAMPLE_FACTOR).float().mean(dim=-1).long()

            # Calculate loss
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            # Use the original, full-length yb as the target
            targets_flat = yb.view(B * T) #<- Changed from yb_downsampled
            loss = F.cross_entropy(logits_flat, targets_flat)

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Update the progress bar description with the current loss
            if steps % 10 == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

        # Save the trained model
        print("\n--- Training Complete ---")
        print(f"Saving model to {MODEL_PATH}...")
        torch.save(m.state_dict(), MODEL_PATH)

    # --- Generate text from the trained or loaded model ---
    print("\n--- Generating New Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_output = m.generate(context, max_new_tokens=200, block_size=block_size)
    generated_text = decode(generated_output[0].tolist())
    print(generated_text)


if __name__ == '__main__':
    main()