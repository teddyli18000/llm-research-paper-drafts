"""Build the fineweb small-run dataset from the public GPT-2 tokenized shards."""
from pathlib import Path
import shutil
import numpy as np
from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)
for name in ("fineweb_train_000001.bin", "fineweb_val_000000.bin"):
    path = hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", repo_type="dataset", filename=name)
    shutil.copyfile(path, DATA / name)

shard = DATA / "fineweb_train_000001.bin"
header = np.fromfile(shard, dtype="<i4", count=256)
assert header[0] == 20240520 and header[1] == 1
tr = np.memmap(shard, dtype="<u2", mode="r", offset=1024)
counts = np.bincount(tr, minlength=65536)
np.save(DATA / "counts.npy", counts)
print(f"Source=kjj0/fineweb10B-gpt2; train shard1 tokens={len(tr)}; counts from same shard")
