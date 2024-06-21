from pathlib import Path

import datasets
import pandas as pd

data = datasets.load_from_disk(str(Path().cwd() / "datasets_cache/wmt/wmt14"))
print(data)

pd.DataFrame(data["train"])
pd.DataFrame(data["test"])
pd.DataFrame(data["validation"])
