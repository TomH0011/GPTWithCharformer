import torch

with open("Sample_Text.txt", 'r') as f:
    text = f.read()  # now it's a string


# import os
#
# # Path to your root folder that contains all subfolders with text files
# root_folder = "TEXT_FOLDERS"
#
# # Output file
# output_file = "Sample_Text.txt"
#
# with open(output_file, "w", encoding="utf-8") as training_data:  # "w" = overwrite each run
#     for root, dirs, files in os.walk(root_folder):
#         for filename in files:
#             if filename.endswith(".txt"):  # only process .txt files
#                 file_path = os.path.join(root, filename)
#                 try:
#                     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                         text = f.read()
#                         training_data.write(text + "\n")  # add newline so files don’t merge
#                         print(f"Processed: {file_path}")
#                 except Exception as e:
#                     print(f"Skipping {file_path} due to error: {e}")



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
