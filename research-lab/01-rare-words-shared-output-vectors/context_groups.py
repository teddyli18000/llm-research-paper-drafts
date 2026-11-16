"""A data-matched static semantic grouping from the actual training tokens."""
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

folder = Path(__file__).resolve().parent / "data"
counts = np.load(folder / "counts.npy")
V = 6000
rank = np.argsort(-counts, kind="stable")
keep = np.unique(np.concatenate([rank[:V - 2], [50256]]))
UNK = len(keep)
look = np.full(65536, UNK, dtype=np.int64)
look[keep] = np.arange(len(keep))
tokens = np.memmap(folder / "fineweb_train_000001.bin", dtype="<u2", mode="r", offset=1024)
ids = look[np.asarray(tokens[:400 * 32 * 64]).astype(np.int64)]
left, right = ids[:-1], ids[1:]
mask = (left != UNK) & (right != UNK)
a, b = left[mask], right[mask]
row = np.concatenate([a, b])
col = np.concatenate([b, a])
cooc = coo_matrix((np.ones(len(row),dtype=np.float32),(row,col)),shape=(V,V)).tocsr()
cooc.sum_duplicates()
features = TfidfTransformer(sublinear_tf=True).fit_transform(cooc)
z = normalize(TruncatedSVD(n_components=48,random_state=0,n_iter=5).fit_transform(features))
labels = KMeans(n_clusters=64, random_state=0,n_init=4,max_iter=50).fit_predict(z)
labels = np.concatenate([labels[:UNK], [0]]).astype(np.int64)
np.save(folder / "context_groups_6000.npy", labels)
print("Context-only static grouping: positions",len(ids),"pairs",len(a),"group min/max",np.bincount(labels,minlength=64).min(),np.bincount(labels,minlength=64).max())
