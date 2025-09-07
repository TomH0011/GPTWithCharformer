import torch
import torch.nn as nn
import torch.nn.functional as F
from charformer_pytorch import GBST
from einops import rearrange


class FullCharformerModel(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, gbst_blocks, gbst_downsample_factor):
        super().__init__()
        self.downsample_factor = gbst_downsample_factor
        self.dim = dim  # Store model dimension

        self.gbst_stem = GBST(
            num_tokens=num_tokens,
            dim=dim,
            blocks=gbst_blocks,
            downsample_factor=gbst_downsample_factor,
            score_consensus_attn=True
        )

        # Use a TransformerDecoderLayer and TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )
        self.transformer_body = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=depth
        )

        self.upsampler = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        )
        self.to_logits = nn.Linear(dim, num_tokens)

    def _generate_square_subsequent_mask(self, sz, device):
        """Generates a square mask for causality."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, mask=None):
        # 1. Downsample with GBST
        x, mask = self.gbst_stem(x, mask=mask)

        # 2. Prepare masks for the decoder
        # The padding mask from GBST needs to be inverted for the PyTorch transformer
        padding_mask = ~mask if mask is not None else None

        # Create a causal mask to prevent the model from looking ahead
        seq_len = x.shape[1]
        causal_mask = self._generate_square_subsequent_mask(seq_len, x.device)

        # 3. Pass through the Transformer Decoder
        # In a decoder-only setup, the input `x` is used for both `tgt` and `memory`
        x = self.transformer_body(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_key_padding_mask=padding_mask
        )

        # 4. Upsample to original sequence length
        x = x.permute(0, 2, 1)
        x = self.upsampler(x)
        x = x.permute(0, 2, 1)

        # 5. Project to logits
        return self.to_logits(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size):
        self.eval()  # Set model to evaluation mode
        # idx is the current context of token indices, shape (B, T)
        for _ in range(max_new_tokens):
            # Crop the context to the last `block_size` tokens
            idx_cond = idx[:, -block_size:]

            # Get the model's predictions
            logits = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the newly sampled token
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()  # Set model back to training mode
        return idx