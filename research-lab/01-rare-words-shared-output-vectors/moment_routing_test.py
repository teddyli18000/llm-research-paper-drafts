"""Analytic checks for projected-update/raw-second-moment Ember routing."""
import torch


def project(g, groups, k):
    sums = torch.zeros(k, g.shape[1], dtype=g.dtype).index_add_(0, groups, g)
    counts = torch.bincount(groups, minlength=k).to(g.dtype)
    return sums[groups] / counts[groups, None]


def factored_second_moment(g, beta2=0.95, steps=1):
    row = g.square().mean(dim=1)
    col = g.square().sum(dim=0) / g.shape[0]
    total = g.square().mean()
    bc = 1 - beta2 ** steps
    return row / bc, col / bc, total / bc


torch.manual_seed(5201)
vocab, width, k = 97, 24, 8
groups = torch.arange(vocab) % k
raw = torch.randn(vocab, width)
raw -= raw.mean(dim=0, keepdim=True)
praw = project(raw, groups, k)
projected = raw + 2.0 * praw

assert torch.allclose(project(projected, groups, k), project(projected, groups, k))
assert torch.allclose(praw.sum(dim=0), torch.zeros(width), atol=1e-5)
assert torch.allclose(project(praw, groups, k), praw, atol=1e-6)

raw_stats = factored_second_moment(raw)
projected_stats = factored_second_moment(projected)
routed_stats = factored_second_moment(raw)
for a, b in zip(raw_stats, routed_stats):
    assert torch.equal(a, b)
assert any(not torch.allclose(a, b) for a, b in zip(raw_stats, projected_stats))

row, col, total = raw_stats
v_hat = row[:, None] * col[None, :] / total
decoupled_update = projected / (v_hat.sqrt() + 1e-8)
standard_row, standard_col, standard_total = projected_stats
standard_v = standard_row[:, None] * standard_col[None, :] / standard_total
standard_update = projected / (standard_v.sqrt() + 1e-8)
assert not torch.allclose(decoupled_update, standard_update)

print(
    "MOMENT_ROUTING_IDENTITY PASS "
    f"projected_to_raw_grad_norm={projected.norm().item()/raw.norm().item():.6f} "
    f"decoupled_to_shared_update_norm={decoupled_update.norm().item()/standard_update.norm().item():.6f}"
)
