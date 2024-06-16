import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import (
    FeedForward,
    InputBlock,
    InputEmbedding,
    MultiHeadAttention,
    PositionalEncoding,
)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_hidden_size: int = 2048,
        trg_mask: torch.tensor = None,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            vocab_size, embed_size, num_heads, dropout, trg_mask
        )
        self.ffn = FeedForward(embed_size, ffn_hidden_size, dropout)
        self.layernorm = nn.LayerNorm(embed_size)


class MaskedMutiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        dropout: float,
        trg_mask: torch.tensor,
    ):
        super().__init__(vocab_size, embed_size, num_heads, dropout)
        self.trg_mask = trg_mask

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len, embed_size]
        batch_size, seq_len = input_x.shape[0], input_x.shape[1]
        key = self.linear_key(input_x).reshape((batch_size, seq_len, self.num_heads, -1))
        query = self.linear_query(input_x).reshape(
            (batch_size, seq_len, self.num_heads, -1)
        )
        value = self.linear_value(input_x).reshape(
            (batch_size, seq_len, self.num_heads, -1)
        )
        attention = (
            key.permute(0, 2, 1, 3) @ query.permute(0, 2, 3, 1) / self.embed_size**0.5
        ) + self.trg_mask  # [batch_size, num_heads, seq_len, seq_len]
        out = (F.softmax(attention, dim=-1) @ value.permute(0, 2, 1, 3)).reshape(
            (batch_size, seq_len, -1)
        )  # out: [batch_size, seq_len, embed_size]
        return self.dropout(out)
