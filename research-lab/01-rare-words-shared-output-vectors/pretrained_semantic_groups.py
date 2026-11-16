"""Strong fixed-group baseline from a pretrained GPT-2 output head."""
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

root = Path(__file__).resolve().parent
data = root / "data"
counts = np.load(data / "counts.npy")
V = 6000
rank = np.argsort(-counts, kind="stable")
keep = np.unique(np.concatenate([rank[:V - 2], [50256]]))
path = hf_hub_download("openai-community/gpt2", "model.safetensors")
with safe_open(path, framework="pt", device="cpu") as handle:
    vectors = handle.get_tensor("transformer.wte.weight").numpy()
X = normalize(vectors[keep])
clusters = KMeans(n_clusters=64, random_state=0, n_init=4, max_iter=50).fit_predict(X)
labels = np.concatenate([clusters, [0]]).astype(np.int64)
np.save(data / "pretrained_groups_6000.npy", labels)
size = np.bincount(labels, minlength=64)
print("GPT-2 semantic output-head clusters",len(labels),"min",size.min(),"max",size.max())
