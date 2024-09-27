import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import FeedForward, InputBlock, MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30000,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int = 2048,
        trg_mask: torch.tensor = None,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.masked_multi_head_attention = MaskedMutiHeadAttention(
            vocab_size, embed_size, num_heads, dropout, trg_mask, device=device
        ).to(device)
        self.cross_attention = DecoderMultiHeadAttention(
            vocab_size,
            embed_size,
            num_heads,
            dropout,
        )
        self.ffn = FeedForward(embed_size, ffn_hidden_size, dropout, device=device)
        self.layernorm = nn.LayerNorm(embed_size, device=device)

    def forward(
        self,
        input_x: torch.tensor,
        x_raw: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
    ):
        # input_x: [batch_size, seq_len, embed_size]
        # key和value来源于Encoder的最终输出
        out_1 = self.layernorm(self.masked_multi_head_attention(input_x) + input_x)
        out_2 = self.layernorm(self.cross_attention(out_1, key, value) + out_1)
        out_3 = self.layernorm(self.ffn(out_2) + out_2)
        return out_3


class MaskedMutiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        dropout: float,
        trg_mask_flag: bool = True,
    ):
        super().__init__(vocab_size, embed_size, num_heads, dropout)
        self.trg_mask_flag = trg_mask_flag
        self.linear_key, self.linear_query, self.linear_value = (
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size),
        )

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len, embed_size]
        batch_size, seq_len = input_x.shape[0], input_x.shape[1]
        attention_mask = torch.where(
            torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)), 0, -1e20
        ).to(self.device)
        key = self.linear_key(input_x).reshape((batch_size, seq_len, self.num_heads, -1))
        query = self.linear_query(input_x).reshape(
            (batch_size, seq_len, self.num_heads, -1)
        )
        value = self.linear_value(input_x).reshape(
            (batch_size, seq_len, self.num_heads, -1)
        )
        attention = (
            key.permute(0, 2, 1, 3) @ query.permute(0, 2, 3, 1) / self.embed_size**0.5
        ) + attention_mask  # [batch_size, num_heads, seq_len, seq_len]
        out = (F.softmax(attention, dim=-1) @ value.permute(0, 2, 1, 3)).reshape(
            (batch_size, seq_len, -1)
        )  # out: [batch_size, seq_len, embed_size]
        return self.dropout(out)


class DecoderMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__(vocab_size, embed_size, num_heads, dropout)

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor):
        # query来自Decoder, key和value来自Encoder
        batch_size, seq_len = query.shape[0], query.shape[1]
        key = key.reshape((batch_size, seq_len, self.num_heads, -1))
        value = value.reshape((batch_size, seq_len, self.num_heads, -1))
        query = query.reshape((batch_size, seq_len, self.num_heads, -1))

        attention = (key.permute(0, 2, 1, 3) @ query.permute(0, 2, 3, 1)) / (
            self.embed_size**0.5
        )
        out = (F.softmax(attention, dim=-1) @ value.permute(0, 2, 1, 3)).reshape(
            (batch_size, seq_len, -1)
        )
        return self.dropout(out)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int = 2048,
        max_seq_len: int = 64,
        trg_mask: bool = False,
        block_num: int = 6,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropput = dropout
        self.fnn_hidden_size = ffn_hidden_size
        self.max_seq_len = max_seq_len
        self.block_num = block_num
        self.dropout = dropout
        self.ffn_hidden_size = ffn_hidden_size
        self.trg_mask = trg_mask
        self.device = device
        self.input_block = InputBlock(vocab_size, embed_size, device)

    def gen_trg_mask(self, trg, trg_padding_idx: int = 0):
        trg_mask = torch.where((trg != trg_padding_idx), 1, 0).unsqueeze(1).unsqueeze(2)
        return trg_mask

    def forward(
        self, trg: torch.tensor, encoder_key: torch.tensor, encoder_value: torch.tensor
    ):
        if trg.shape[1] <= self.max_seq_len:
            trg = F.pad(trg, (0, self.max_seq_len - trg.shape[1]))
        else:
            trg = trg[..., : self.max_seq_len]
        if self.trg_mask:
            trg_mask = self.gen_trg_mask(trg)
        else:
            trg_mask = None
        self.decoder_block = DecoderBlock(
            self.vocab_size,
            self.embed_size,
            self.num_heads,
            self.dropout,
            trg_mask=trg_mask,
            device=self.device,
        )
        self.decoder = nn.ModuleList([self.decoder_block for _ in range(self.block_num)])
        x = self.input_block(trg)
        for layer in self.decoder:
            x = layer(x, encoder_key, encoder_value)
        return x
