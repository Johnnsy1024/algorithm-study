import torch

batchsize = 2
seq_len = 4
num_heads = 3

padding_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])  # Shape: [batchsize, seq_len]
padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: [batchsize, 1, 1, seq_len]
print(padding_mask)
