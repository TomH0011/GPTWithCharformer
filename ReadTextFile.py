import torch

with open("Sample_Text.txt", 'r') as f:
    text = f.read()  # now it's a string

# Create encoder and decoder
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create an id for each char for charformer to use
str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [str_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_str[i] for i in l])

# Wrap it into a tensor of longs
data = torch.tensor(encode(text), dtype=torch.long)

# Split into validation and training sets in a 10:90 split
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]
