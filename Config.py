import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'using device: {device}')

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

batch_size = 256
block_size = 63
learning_rate = 0.0001
max_iters = 10000
eval_interval = 500
MODEL_DIM = 256
DOWNSAMPLE_FACTOR = 3  # block size % DOWNSAMPLE_FACTOR must equal 0
depth = 6  # Number of attention layers
heads = 8  # Number of attention heads looking for patterns
gbst_blocks = ((3, 0), (3, 1), (3, 2))
weight_decay = 0.1
dropout = 0.1
