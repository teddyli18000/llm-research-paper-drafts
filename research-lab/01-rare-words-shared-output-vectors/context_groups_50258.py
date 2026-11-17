"""Construct context-only static token groups at near-full GPT-2 vocabulary."""
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

t0 = time.monotonic()
data = Path(__file__).resolve().parent / "data"
counts = np.load(data / "counts.npy")
V_ARG = 50258
rank = np.argsort(-counts, kind="stable")
keep = np.unique(np.concatenate([rank[:V_ARG-2], [50256]]))
vocab = len(keep) + 1
unk = vocab - 1
look = np.full(65536, unk, dtype=np.int64)
look[keep] = np.arange(len(keep))
train = np.memmap(data / "fineweb_train_000001.bin", dtype="<u2", mode="r", offset=1024)
tokens = 800 * 32 * 64
ids = look[np.asarray(train[:tokens]).astype(np.int64)]
prev_ids, next_ids = ids[:-1], ids[1:]
valid = (prev_ids != unk) & (next_ids != unk)
a, b = prev_ids[valid], next_ids[valid]
src = np.concatenate((a, b))
dst = np.concatenate((b, a))
cooc = coo_matrix(
    (np.ones(len(src), dtype=np.float32), (src, dst)),
    shape=(vocab, vocab),
).tocsr()
cooc.sum_duplicates()
features = TfidfTransformer(sublinear_tf=True).fit_transform(cooc)
embed = TruncatedSVD(n_components=48, random_state=0, n_iter=5).fit_transform(features)
normed = normalize(embed[:unk])
labels = KMeans(n_clusters=64, random_state=0, n_init=4, max_iter=50).fit_predict(normed)
labels = np.concatenate((labels, [0])).astype(np.int64)
assert len(labels) == vocab
np.save(data / "context_groups_50258.npy", labels)
bincount = np.bincount(labels, minlength=64)
unseen = np.bincount(ids, minlength=vocab)
print(
    f"STATIC_CONTEXT_50K tokens={tokens} vocab={vocab} "
    f"observed_rows={(unseen[:unk]>0).sum()} "
    f"unseen_rows={(unseen[:unk]==0).sum()} "
    f"cooccurrence_nnz={cooc.nnz} "
    f"min_group={bincount.min()} max_group={bincount.max()} "
    f"preprocessing_seconds={time.monotonic()-t0:.1f}",
    flush=True,
)
