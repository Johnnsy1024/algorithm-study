import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        max_seq_len: int = 128,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embed_size, device=device)
        self.max_seq_len = max_seq_len

    def forward(self, input_x: torch.tensor):  # input_x: [batch_size, seq_len]
        out = self.input_embedding(input_x)  # out: [batch_size, seq_len, embed_size]
        return out


class PositionalEncoding:
    def __init__(self, embed_size: int, max_seq_len: int = 128):
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len

    def __call__(self, input_x: torch.tensor):
        # if input_x.shape[1] <= self.max_seq_len:
        #     input_x = F.pad(input_x, (0, self.max_seq_len - input_x.shape[1]))
        # else:
        #     input_x = input_x[..., self.max_seq_len]
        return self._positional_encoding_vector(
            input_x, self.embed_size
        )  # [batch_size, seq_len, embed_size]

    def _positional_encoding_vector(self, input_x: torch.tensor, embed_size: int):
        batch_size, seq_len = input_x.shape[0], input_x.shape[1]
        pos_encoding = torch.zeros(batch_size, seq_len, embed_size)
        pos_encoding_element = torch.arange(0, seq_len)[..., None].tile(
            (1, embed_size)
        ) / (
            10000
            ** ((torch.arange(embed_size) // 2) / embed_size)[None, ...].tile(seq_len, 1)
        )
        pos_encoding[..., 0::2] = torch.sin(pos_encoding_element[..., 0::2])
        pos_encoding[..., 1::2] = torch.cos(pos_encoding_element[..., 1::2])

        return pos_encoding

    def _positional_encoding(self, input_x: torch.tensor, embed_size: int):
        batch_size, seq_len = input_x.shape[0], input_x.shape[1]
        pos_encoding = torch.zeros(batch_size, seq_len, embed_size)
        for i in range(embed_size):
            if i % 2 == 0:
                pos_encoding[:, :, i] = torch.sin(
                    torch.arange(0, seq_len) / 10000 ** (i / embed_size)
                )
            else:
                pos_encoding[:, :, i] = torch.cos(
                    torch.arange(0, seq_len) / 10000 ** ((i - 1) / embed_size)
                )
        return pos_encoding


class InputBlock(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, device: torch.device = "cpu"):
        super().__init__()
        self.device = device
        self.input_embedding = InputEmbedding(vocab_size, embed_size, device=device)
        self.positional_embedding = PositionalEncoding(embed_size)
        self.device = device

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len]
        x = self.input_embedding(input_x) + self.positional_embedding(input_x).to(
            self.device
        )

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        src_mask: torch.tensor = None,
        device: str = "cpu",
    ):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.src_mask = src_mask
        self.input_embedding = InputEmbedding(vocab_size, embed_size, device).to(device)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.linear_key = nn.Linear(embed_size, embed_size, device=device)
        self.linear_query = nn.Linear(embed_size, embed_size, device=device)
        self.linear_value = nn.Linear(embed_size, embed_size, device=device)
        self.dropout = nn.Dropout(dropout)

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
        )  # [batch_size, num_heads, seq_len, seq_len]
        if self.src_mask is not None:
            attention = attention + self.src_mask
        out = (F.softmax(attention, dim=-1) @ value.permute(0, 2, 1, 3)).reshape(
            (batch_size, seq_len, -1)
        )  # out: [batch_size, seq_len, embed_size]
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_size: int = 512,
        ffn_hidden_size: int = 2048,
        dropout: float = 0.1,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.linear_1 = nn.Linear(embed_size, ffn_hidden_size, device=device)
        self.linear_2 = nn.Linear(ffn_hidden_size, embed_size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len ,embed_size]
        x = self.linear_1(input_x)
        x = F.relu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int = 2048,
        src_mask: torch.tensor = None,
        device: str = "cpu",
    ):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.multi_head_attention = MultiHeadAttention(
            vocab_size, embed_size, num_heads, dropout, src_mask, device
        )
        self.ffn = FeedForward(embed_size, ffn_hidden_size, dropout, device)
        self.layernorm = nn.LayerNorm(embed_size, device=device)

    def forward(
        self, input_x: torch.tensor
    ):  # input_x: [batch_size, seq_len, embed_size]
        out_1 = self.layernorm(self.multi_head_attention(input_x) + input_x)
        out_2 = self.layernorm(self.ffn(out_1) + out_1)
        return out_2


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 120,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int = 2048,
        max_seq_len: int = 64,
        src_mask_flag: bool = True,
        block_num: int = 6,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.src_mask_flag = src_mask_flag
        self.num_heads = num_heads
        self.dropput = dropout
        self.fnn_hidden_size = ffn_hidden_size
        self.max_seq_len = max_seq_len
        self.block_num = block_num
        self.dropout = dropout
        self.ffn_hidden_size = ffn_hidden_size
        self.device = device

        self.input_block = InputBlock(vocab_size, embed_size, device)

    def gen_src_mask(self, src, src_padding_idx: int = 0):
        # src: [batch_size, seq_len]
        src_mask = (
            torch.where((src != src_padding_idx), 0, -1e20).unsqueeze(1).unsqueeze(1)
        )  # [batch_size, 1, 1, seq_len]如果只保持[batch_size, seq_len]的形状的话，
        # 无法与[batch_size, num_heads, seq_len, seq_len]形状的attention进行广播
        return src_mask

    def forward(self, input_x: torch.tensor):
        # input_x: [batch_size, seq_len]
        if input_x.shape[1] <= self.max_seq_len:
            input_x = F.pad(input_x, (0, self.max_seq_len - input_x.shape[1]))
        else:
            input_x = input_x[..., : self.max_seq_len]
        if self.src_mask_flag:
            src_mask = self.gen_src_mask(input_x)
        else:
            src_mask = None

        self.encoder_block = EncoderBlock(
            self.vocab_size,
            self.embed_size,
            self.num_heads,
            self.dropout,
            self.ffn_hidden_size,
            src_mask,
            device=self.device,
        )
        self.encoder = nn.ModuleList([self.encoder_block for _ in range(self.block_num)])
        x = self.input_block(input_x)
        for layer in self.encoder:
            x = layer(x)
        return x


if __name__ == "__main__":
    # pass
    device = torch.device("mps")
    input_x = torch.randint(low=0, high=32, size=(32, 32), device=device)
    encoder = Encoder(vocab_size=32, src_mask_flag=True, device=device)
    print(encoder(input_x).shape)
    pass
