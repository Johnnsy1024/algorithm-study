# %%
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_ZH_SRC_LEN = 64
MAX_EN_TRG_LEN = 80

# %%
# 加载原词库
ds = load_dataset("Aye10032/zh-en-translate-20k")

ds_train = ds["train"].to_pandas()
ds_test = ds["validation"].to_pandas()

ds_train_zh = ds_train["chinese"]
ds_train_en = ds_train["english"]

ds_test_zh = ds_test["chinese"]
ds_test_en = ds_test["english"]

ds_zh = pd.concat([ds_train_zh, ds_test_zh])
ds_en = pd.concat([ds_train_en, ds_test_en])
# %%
# 加载中文分词器

tokenizer_zh = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
# 添加句首和句尾的special token
tokenizer_zh.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})
# 根据现有语料训练分词器
tokenizer_zh.train_new_from_iterator(ds_zh.tolist(), vocab_size=30000)

# 加载英文分词器
tokenizer_en = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
tokenizer_en.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})
tokenizer_en.train_new_from_iterator(ds_en.tolist(), vocab_size=30000)
# %%
# 训练数据分词
train_raw_data_src = tokenizer_zh(
    ds_train_zh.tolist(),
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_ZH_SRC_LEN,
    return_attention_mask=True,
)
train_raw_data_trg = tokenizer_en(
    ds_train_en.tolist(),
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_EN_TRG_LEN,
    return_attention_mask=True,
)
# 测试集数据分词
test_raw_data_src = tokenizer_zh(
    ds_test_zh.tolist(),
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_ZH_SRC_LEN,
    return_attention_mask=True,
)
test_raw_data_trg = tokenizer_en(
    ds_test_en.tolist(),
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_EN_TRG_LEN,
    return_attention_mask=True,
)

# %%
