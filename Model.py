import torch
import torch.nn as nn
import torch.nn.functional as F
from charformer_pytorch import GBST
from Config import dropout
from einops import rearrange
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ImprovedCharformerModel(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, gbst_blocks, gbst_downsample_factor):
        super().__init__()
        self.downsample_factor = gbst_downsample_factor
        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        # GBST for learned tokenisation
        self.gbst_stem = GBST(
            num_tokens=num_tokens,
            dim=dim,
            blocks=gbst_blocks,
            downsample_factor=gbst_downsample_factor,
            score_consensus_attn=True
        )

        # Add positional encoding for better position awareness
        self.pos_encoder = PositionalEncoding(dim)

        # Use standard transformer decoder layers properly
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,  # Standard FFN size
                dropout=dropout,
                activation='gelu',  # Better than relu for language modeling
                batch_first=True,
                norm_first=True  # Pre-norm architecture, more stable
            ) for _ in range(depth)
        ])

        # Layer norm after transformer
        self.ln_f = nn.LayerNorm(dim)

        # Upsampling with residual connection
        self.upsampler = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=gbst_downsample_factor,
            stride=gbst_downsample_factor
        )

        # Output projection with better initialisation
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        # Initialise output projection with small weights
        nn.init.normal_(self.to_logits.weight, mean=0.0, std=0.02)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        # 1. Downsample with GBST
        x, mask = self.gbst_stem(x, mask=mask)

        # 2. Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # 3. Create causal mask for autoregressive modeling
        seq_len_down = x.shape[1]
        causal_mask = self._generate_square_subsequent_mask(seq_len_down, x.device)

        # 4. Pass through transformer layers with proper masking
        for layer in self.transformer_layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=~mask if mask is not None else None)

        # 5. Final layer norm
        x = self.ln_f(x)

        # 6. Upsample to original sequence length
        x = x.permute(0, 2, 1)
        x = self.upsampler(x)
        x = x.permute(0, 2, 1)

        # Ensure we have the right sequence length
        if x.shape[1] != seq_len:
            x = x[:, :seq_len]  # Truncate if necessary

        # 7. Project to vocabulary
        logits = self.to_logits(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size, temperature=1.0, top_k=50, top_p=0.95):
        """Enhanced generation with top-k and top-p sampling"""
        # top-k and top-p to avoid picking unlikely tokens
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:] if idx.shape[1] > block_size else idx

            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx


class SimpleTransformerModel(nn.Module):
    """Alternative: A simpler, more standard transformer for comparison"""

    def __init__(self, num_tokens, dim, depth, heads, block_size, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.block_size = block_size

        # Token and position embeddings
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)

        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, top_p=0.95):
        """Generate with top-k and top-p sampling"""
        # again, top-k and top-p to avoid picking unlikely tokens
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.block_size:] if idx.shape[1] > self.block_size else idx

            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx