from typing import List

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset


def extract_raw_data(data_name: str = "Aye10032/zh-en-translate-20k"):
    ds = load_dataset(data_name)
    ds_train = ds["train"].to_pandas()
    ds_test = ds["validation"].to_pandas()

    return ds_train, ds_test


def tokenizer_train(corpus: List[str]):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=5000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[BOS]", "[EOS]", "[MASK]"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer


def get_input(tokenizer: Tokenizer, raw_data: List[str]):
    encoding_list = tokenizer.encode_batch(raw_data)
    res = []
    for token in encoding_list:
        res.append(token.ids)
    return res


class TranslateDataset(Dataset):
    def __init__(
        self,
        src: List[int],
        trg: List[int],
        src_max_length: int,
        trg_max_length: int,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length
        self.trg_max_length = trg_max_length
        self.src = src
        self.trg = trg

    def __getitem__(self, index: int):
        src = self.src[index]
        trg = self.trg[index]
        # 向目标序列的句首添加[BOS]，句尾添加[EOS]
        trg.insert(0, self.tokenizer.encode("[BOS]").ids[0])
        trg.append(self.tokenizer.encode("[EOS]").ids[0])
        if len(src) > self.src_max_length:
            src = src[: self.src_max_length]
        if len(trg) > self.trg_max_length:
            trg = trg[: self.trg_max_length]

        while len(src) < self.src_max_length:
            src.append(self.tokenizer.encode("[PAD]").ids[0])
        while len(trg) < self.trg_max_length:
            trg.append(self.tokenizer.encode("[PAD]").ids[0])

        return torch.tensor(src), torch.tensor(trg)

    def __len__(self):
        return len(self.src)


if __name__ == "__main__":
    corpus = extract_raw_data()
    tokenizer = tokenizer_train(corpus)
    res = get_input(tokenizer, corpus)
    pass
