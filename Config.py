import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

# TRAINING HYPERPARAMETERS

# Batch and sequence settings
batch_size = 32  # Reduced for stability
block_size_iter = 84  # Adjusted for better divisibility
DOWNSAMPLE_FACTOR = 4  # Standard downsampling
block_size = DOWNSAMPLE_FACTOR * block_size_iter  # 336 tokens context

# Learning rate settings
learning_rate = 3e-4  # Standard for transformers
min_learning_rate = 1e-6  # For cosine annealing
weight_decay = 0.01  # Lower weight decay

# Training duration
max_iters = 1000000  # Much more training needed for character-level
eval_interval = 500  # More frequent evaluation
eval_iters = 40  # More evaluation batches for stable metrics

# Model architecture
MODEL_DIM = 512  # Keep this
depth = 8  # Number of transformer layers
heads = 8  # Number of attention heads
dropout = 0.1  # Reduced dropout for better learning

# GBST settings (for Charformer)
gbst_blocks = ((2, 0), (3, 0), (4, 0), (5, 0), (6, 0))  # More variety in block sizes

# Generation settings
temperature = 0.8  # Balanced creativity
top_k = 50  # Top-k sampling
top_p = 0.95  # Nucleus sampling
MAX_NEW_TOKENS = 500  # Generation length

# Training optimisation
accumulation_steps = 4  # Gradient accumulation for larger effective batch
gradient_clip = 1.0  # Gradient clipping value



print("TRAINING RECOMMENDATIONS")

print("""
For 353 MB of text data (character-level) (adjust based on your text data set size):

1. MINIMUM Training Requirements:
   - Iterations: 200,000-500,000 (currently set to {})
   - Time estimate: 10-20 hours on GPU

2. Data Quality Check:
   - Ensure your text file is properly encoded (UTF-8)
   - Remove any corrupted characters or encoding issues
   - Consider cleaning: removing excessive whitespace, normalising quotes, etc.

3. Monitoring:
   - Val loss should decrease steadily
   - If val loss plateaus while train loss decreases = overfitting
   - Generate samples every 5000 steps to check quality
""".format(max_iters))

print("="*50)