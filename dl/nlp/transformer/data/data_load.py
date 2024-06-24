from pathlib import Path

import datasets

cur_dir = str(Path().cwd())
if cur_dir.split("/")[-1] != "data":
    cur_dir += "/data"
data = datasets.load_from_disk(cur_dir + "/datasets_cache/wmt/wmt14")

train_data_raw = data["train"]
test_data_raw = data["test"]
valid_data_raw = data["validation"]
