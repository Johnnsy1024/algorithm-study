from pathlib import Path

from datasets import load_dataset

cur_dir = Path().cwd()
cache_dir = cur_dir / "datasets_cache"
dataset = load_dataset("wmt/wmt14", "de-en", cache_dir=str(cache_dir))
dataset.save_to_disk(str(cache_dir / "wmt/wmt14"))
