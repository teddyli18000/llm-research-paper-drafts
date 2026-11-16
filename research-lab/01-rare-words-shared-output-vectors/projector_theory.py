"""Exact centering properties of size-normalized group-gradient projectors."""
import torch
import torch.nn.functional as F

torch.manual_seed(88)
V, K, d = 21, 4, 7
labels = torch.repeat_interleave(torch.arange(K), torch.tensor([2, 4, 6, 9]))
A = F.one_hot(labels, K).float()
Dinv = torch.diag(1.0 / torch.bincount(labels, minlength=K).float())
P = A @ Dinv @ A.T

assert torch.allclose(P, P.T, atol=1e-7)
assert torch.allclose(P @ P, P, atol=1e-6)
assert torch.allclose(P @ torch.ones(V), torch.ones(V), atol=1e-6)

G = torch.randn(V, d)
G -= G.mean(0)
assert torch.allclose(G.sum(0), torch.zeros(d), atol=1e-6)
assert torch.allclose((P @ G).sum(0), torch.zeros(d), atol=1e-5)
unnormalized = (A @ A.T) @ G
assert unnormalized.sum(0).norm() > 1.0

eta = 0.1
lam = 1.2
update = -eta * (G + lam * P @ G)
assert torch.allclose(update.sum(0), torch.zeros(d), atol=1e-5)
print("PASS group projector: symmetric, idempotent, constant eigenvector, preserves centered output-head gradients")
print("Unnormalized AAᵀ caused mean drift L2", unnormalized.sum(0).norm().item())
