from typing import List, Literal

import pandas as pd
import torch

# import torch.utils
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import BertPreTokenizer, Whitespace
from tokenizers.trainers import BpeTrainer


def extract_raw_data(data_name: str = "Aye10032/zh-en-translate-20k"):
    ds = load_dataset(data_name)
    ds_train = ds["train"].to_pandas()
    ds_test = ds["validation"].to_pandas()

    return ds_train, ds_test


def tokenizer_train(corpus: List[str]):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=30000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer


def get_input(tokenizer: Tokenizer, raw_data: List[str]):
    encoding_list = tokenizer.encode_batch(raw_data)
    tokenizer.encode()
    res = []
    for token in encoding_list:
        res.append(token.ids)
    return res


class TranslateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_name: str = "Aye10032/zh-en-translate-20k",
        src_max_length: int = 128,
        trg_max_length: int = 128,
        dateset_type: Literal["train", "eval", "test"] = "train",
        device: torch.device = "cpu",
    ) -> None:
        super().__init__()
        assert dateset_type in [
            "train",
            "eval",
            "test",
        ], "数据集类型必须为train/eval/test"
        assert (
            data_name == "Aye10032/zh-en-translate-20k"
        ), "暂时只支持Aye10032/zh-en-translate-20k数据集"
        self.data_name = data_name
        self.src_max_length = src_max_length
        self.trg_max_length = trg_max_length
        self.dataset_type = dateset_type
        self.device = device

        self.src, self.trg = self._prepare_dataset()

    def _prepare_dataset(self):
        # 提取原始翻译语句
        ds = load_dataset(self.data_name)
        if len(ds) > 1:
            res = []
            for d in ds.values():
                res.append(d.to_pandas())
            ds = pd.concat(res)
        train_eval_ds, test_ds = train_test_split(ds, test_size=0.2)
        train_ds, eval_ds = train_test_split(train_eval_ds, test_size=0.1)
        train_src, train_trg, eval_src, eval_trg, test_src, test_trg = (
            train_ds.iloc[:, 0].to_list(),
            train_ds.iloc[:, 1].to_list(),
            eval_ds.iloc[:, 0].to_list(),
            eval_ds.iloc[:, 1].to_list(),
            test_ds.iloc[:, 0].to_list(),
            test_ds.iloc[:, 1].to_list(),
        )
        # 构建分词器
        tokenizer = self._train_tokenizer(ds)
        # 分词(未填充or截断)
        if self.dataset_type == "train":
            src, trg = train_src, train_trg
        elif self.dataset_type == "eval":
            src, trg = eval_src, eval_trg
        else:
            src, trg = test_src, test_trg
        src_encoded, trg_encoded = self._encode_corpus(
            src,
            trg,
            tokenizer=tokenizer,
        )

        return self._add_special_tokens(tokenizer, src_encoded), self._add_special_tokens(
            tokenizer, trg_encoded
        )

    def _add_special_tokens(self, tokenizer: Tokenizer, corpus: List[List[int]]):
        for idx in range(len(corpus)):
            corpus[idx].insert(0, tokenizer.encode("[BOS]").ids[0])
            if len(corpus[idx]) >= self.src_max_length:
                corpus[idx] = corpus[idx][: self.src_max_length]
            else:
                while len(corpus[idx]) < self.src_max_length:
                    corpus[idx].append(tokenizer.encode("[PAD]").ids[0])
            corpus[idx][-1] = tokenizer.encode("[EOS]").ids[0]
        return corpus

    def _encode_corpus(
        self,
        *src_trg: List[List[str]],
        tokenizer: Tokenizer = None,
    ):
        src_encoded, trg_encoded = [], []
        src, trg = src_trg
        for i in range(len(src)):
            src_encoded.append(tokenizer.encode(src[i]).ids)
        for i in range(len(trg)):
            trg_encoded.append(tokenizer.encode(trg[i]).ids)
        return src_encoded, trg_encoded

    def _train_tokenizer(
        self, ds: DatasetDict | Dataset | IterableDatasetDict | IterableDataset
    ) -> Tokenizer:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer_trainer = BpeTrainer(
            vocab_size=30000,
            min_frequency=5,
            show_progress=True,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"],
        )
        tokenizer.train_from_iterator(
            ds.iloc[:, 0].to_list() + ds.iloc[:, 1].to_list(), tokenizer_trainer
        )

        return tokenizer

    def __getitem__(self, index: int):

        return torch.tensor(self.src[index], device=self.device), torch.tensor(
            self.trg[index], device=self.device
        )

    def __len__(self):
        return len(self.src)


if __name__ == "__main__":
    device = torch.device("mps")
    trans_dataset = TranslateDataset(device=device)
    pass
