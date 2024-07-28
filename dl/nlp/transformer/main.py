import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder
from tokenizer import (
    MAX_EN_TRG_LEN,
    MAX_ZH_SRC_LEN,
    test_raw_data_src,
    test_raw_data_trg,
    tokenizer_en,
    tokenizer_zh,
    train_raw_data_src,
    train_raw_data_trg,
)
from torch.utils.data import DataLoader

encoder = Encoder(
    vocab_size=tokenizer_zh.vocab_size,
    embed_size=512,
    src_mask_flag=True,
    max_seq_len=MAX_ZH_SRC_LEN,
)
encoder(train_raw_data_src["input_ids"][:100, :])
