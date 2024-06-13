import torch

batchsize = 2
seq_len = 4
num_heads = 4
embed_size = 32
vocab_size = 5

input_x = torch.tensor([[4, 3, 0, 0], [3, 1, 2, 0]])
padding_mask = torch.where(
    (input_x != 0).unsqueeze(1).unsqueeze(2), 0, -1e20
)  # Shape: [batchsize, 1, 1, seq_len]

# embedding层
input_embedding = torch.nn.Embedding(vocab_size, embed_size)

# position embedding层
position_embedding = torch.nn.Embedding(seq_len, embed_size)

# attention层
key_mapping, value_mapping, query_mapping = [
    torch.nn.Linear(embed_size, embed_size) for _ in range(3)
]

# position_embedding + input_embedding
embedded_input = position_embedding(
    torch.arange(0, seq_len).tile((batchsize, 1))
) + input_embedding(input_x)

# pass results above into q\k\v
key, query, value = (
    key_mapping(embedded_input).reshape((batchsize, seq_len, num_heads, -1)),
    query_mapping(embedded_input).reshape((batchsize, seq_len, num_heads, -1)),
    value_mapping(embedded_input).reshape((batchsize, seq_len, num_heads, -1)),
)

# self-attention
attention = torch.matmul(key.permute(0, 2, 1, 3), query.permute(0, 2, 3, 1))

# attention + mask
attention_w_mask = attention + padding_mask

# softmax
attention_w_softmax = torch.nn.functional.softmax(attention_w_mask, dim=-1)
# pass
