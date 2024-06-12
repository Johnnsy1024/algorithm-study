import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
    ):
        self.query = nn.Embedding()
