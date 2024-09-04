import pandas as pd
import torch
import torch.nn as nn
from data_tokenize import TranslateDataset, extract_raw_data, get_input, tokenizer_train
from Decoder import Decoder
from Encoder import Encoder
from loguru import logger
from lr_scheduler2 import TransformerLRScheduler
from torch.utils.data import DataLoader

# 提取原始语料数据
data_train, data_test = extract_raw_data()
# 训练分词器
ds_df = pd.concat([data_train, data_test])
corpus = ds_df["english"].to_list() + ds_df["chinese"].to_list()
tokenizer = tokenizer_train(corpus)
# 提取输入数据
src_train_input = get_input(tokenizer, data_train.chinese.to_list())
trg_train_input = get_input(tokenizer, data_train.english.to_list())

src_test_input = get_input(tokenizer, data_test.chinese.to_list())
trg_test_input = get_input(tokenizer, data_test.english.to_list())

# 生成输入的数据集
train_dataset = TranslateDataset(src_train_input, trg_train_input, 64, 64, tokenizer)
test_dataset = TranslateDataset(src_test_input, trg_test_input, 64, 64, tokenizer)


# 生成DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
# test_dataloader = DataLoader(test_dataset, batch_size=56, shuffle=True)


# 构建完整的Transformer
class Transformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder_k_linear = nn.Linear(
            self.encoder.embed_size, self.encoder.embed_size
        )
        self.encoder_v_linear = nn.Linear(
            self.encoder.embed_size, self.encoder.embed_size
        )
        self.output_linear = nn.Linear(self.encoder.embed_size, self.encoder.vocab_size)
        self.num_heads = self.encoder.num_heads

    def forward(self, src: torch.tensor, trg: torch.tensor):
        src = src.to(device)
        trg = trg.to(device)
        batch_size, seq_len = src.shape[0], src.shape[1]
        encoder_res = self.encoder(
            src
        )  # encoder_res: [batch_size, seq_length, embedding_size]
        encoder_decoder_key = self.encoder_k_linear(encoder_res).reshape(
            (batch_size, seq_len, -1)
        )
        encoder_decoder_value = self.encoder_v_linear(encoder_res).reshape(
            (batch_size, seq_len, -1)
        )
        decoder_res = self.decoder(trg, encoder_decoder_key, encoder_decoder_value)

        output = self.output_linear(decoder_res)  # [batch_size, seq_len, vocab_size]

        return output


vocab_size = tokenizer.get_vocab_size()

device = torch.device("mps:0")

encoder = Encoder(vocab_size, device=device)
decoder = Decoder(vocab_size, device=device)
model = Transformer(encoder, decoder, device)

EPOCH = 10

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerLRScheduler(optimizer, 512, 500)
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for batch_idx, (src, trg) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(src, trg)
        loss = loss_func(
            output.view(
                -1,
                30000,
            ).float(),
            trg.view(-1).to(device),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        logger.info(
            f"Epoch {epoch + 1}, batch {batch_idx + 1}: loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}"
        )

        # scheduler.step()
        # print("Current loss:", loss.item())
        # print("Current lr:", scheduler.get_last_lr())
