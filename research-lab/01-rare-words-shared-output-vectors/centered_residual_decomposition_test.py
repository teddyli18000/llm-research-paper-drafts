"""Verify centered class residuals remove the softmax-invariant vocabulary-uniform mode."""
import torch
import torch.nn.functional as F


def project(g, groups, k):
    sums = torch.zeros(k, g.shape[1], dtype=g.dtype).index_add_(0, groups, g)
    counts = torch.bincount(groups, minlength=k).clamp_min(1).to(g.dtype)
    return sums[groups] / counts[groups, None]


torch.manual_seed(6171)
n, vocab, width, k = 23, 41, 13, 7
groups = torch.arange(vocab) % k
groups = groups[torch.randperm(vocab)]
h = torch.randn(n, width, dtype=torch.float64)
head = torch.randn(vocab, width, dtype=torch.float64)
target = torch.randint(vocab, (n,))
prob = F.softmax(h @ head.T, dim=-1)
residual = prob.clone()
residual[torch.arange(n), target] -= 1
raw = residual.T @ h / n

counts = torch.bincount(groups, minlength=k).clamp_min(1)
group_prob = torch.zeros(n, k, dtype=h.dtype).index_add_(1, groups, prob)
pred_group = group_prob.T @ h / n
pred_corr = pred_group[groups] / counts[groups, None]
target_group = torch.zeros(k, width, dtype=h.dtype).index_add_(0, groups[target], h)
target_corr = target_group[groups] / counts[groups, None] / n

mean_h = h.mean(dim=0)
uniform_corr = mean_h[None, :].expand(vocab, width) / vocab
pred_centered = pred_corr - uniform_corr
target_centered = target_corr - uniform_corr
full = pred_corr - target_corr

assert torch.allclose(project(raw, groups, k), full, atol=1e-12, rtol=1e-10)
assert torch.allclose(pred_centered - target_centered, full, atol=1e-12, rtol=1e-10)
assert torch.allclose(pred_centered.sum(dim=0), torch.zeros(width, dtype=h.dtype), atol=1e-12)
assert torch.allclose(target_centered.sum(dim=0), torch.zeros(width, dtype=h.dtype), atol=1e-12)

alpha = 2.0
logits = h @ head.T
shifted_head = head + alpha * uniform_corr
shifted_logits = h @ shifted_head.T
assert torch.allclose(
    logits.softmax(dim=-1), shifted_logits.softmax(dim=-1), atol=1e-12, rtol=1e-10
)

print(
    "CENTERED_RESIDUAL_IDENTITY PASS "
    f"pred_centered_norm={pred_centered.norm().item():.6f} "
    f"target_centered_norm={target_centered.norm().item():.6f} "
    f"uniform_norm={uniform_corr.norm().item():.6f} "
    f"max_error={(pred_centered-target_centered-full).abs().max().item():.3e}"
)
