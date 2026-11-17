"""Verify the exact decomposition P(p-y)hᵀ = Pp hᵀ - Py hᵀ."""
import torch
import torch.nn.functional as F


def project(g, groups, k):
    sums = torch.zeros(k, g.shape[1], dtype=g.dtype).index_add_(0, groups, g)
    counts = torch.bincount(groups, minlength=k).clamp_min(1).to(g.dtype)
    return sums[groups] / counts[groups, None]


torch.manual_seed(5471)
n, vocab, width, k = 19, 31, 9, 6
groups = torch.tensor([0] * 2 + [1] * 3 + [2] * 7 + [3] * 5 + [4] * 8 + [5] * 6)
h = torch.randn(n, width)
head = torch.randn(vocab, width, requires_grad=True)
targets = torch.randint(vocab, (n,))
logits = F.linear(h, head)
prob = logits.softmax(dim=-1)
loss = F.cross_entropy(logits, targets, reduction="sum") / n
loss.backward()
raw = head.grad.detach()

counts = torch.bincount(groups, minlength=k).clamp_min(1)
group_prob = torch.zeros(n, k).index_add_(1, groups, prob.detach())
pred_group = group_prob.T @ h / n
pred_corr = pred_group[groups] / counts[groups, None]
target_group = torch.zeros(k, width).index_add_(0, groups[targets], h)
target_corr = target_group[groups] / counts[groups, None] / n
full_corr = pred_corr - target_corr
matrix_corr = project(raw, groups, k)

assert torch.allclose(matrix_corr, full_corr, atol=1e-6, rtol=1e-5)
assert torch.allclose(project(matrix_corr, groups, k), matrix_corr, atol=1e-6)
assert torch.allclose(raw.sum(dim=0), torch.zeros(width), atol=1e-6)
assert torch.allclose(full_corr.sum(dim=0), torch.zeros(width), atol=1e-6)
assert pred_corr.norm() > 0 and target_corr.norm() > 0
print(
    "RESIDUAL_DECOMPOSITION_IDENTITY PASS "
    f"pred_norm={pred_corr.norm().item():.6f} "
    f"target_norm={target_corr.norm().item():.6f} "
    f"full_norm={full_corr.norm().item():.6f} "
    f"max_error={(matrix_corr-full_corr).abs().max().item():.3e}"
)
