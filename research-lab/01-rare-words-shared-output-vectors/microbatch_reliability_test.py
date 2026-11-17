"""Analytic checks for the cross-microbatch projected-gradient reliability estimator."""
import torch


def project(g, groups, k):
    sums = torch.zeros(k, g.shape[1], dtype=g.dtype).index_add_(0, groups, g)
    counts = torch.bincount(groups, minlength=k).to(g.dtype)
    return sums[groups] / counts[groups, None]


def reliability(a, b):
    cross = (a * b).sum()
    energy = 0.5 * (a.square().sum() + b.square().sum())
    return cross / energy.clamp_min(1e-30)


torch.manual_seed(1701)
vocab, width, groups_n = 128, 16, 8
groups = torch.arange(vocab) % groups_n
p = torch.randn(vocab, width)
projected = project(p, groups, groups_n)
assert torch.allclose(project(projected, groups, groups_n), projected)
assert -1.000001 <= float(reliability(p, p * 0.7)) <= 1.000001
assert abs(float(reliability(p, p)) - 1.0) < 1e-6
assert abs(float(reliability(p, -p)) + 1.0) < 1e-6

mu = torch.randn(groups_n, width)[groups]
projected_signal = project(mu, groups, groups_n)
signal_energy = projected_signal.square().sum()
cross_sum = 0.0
energy_sum = 0.0
trials = 2000
for _ in range(trials):
    g1 = mu + torch.randn(vocab, width)
    g2 = mu + torch.randn(vocab, width)
    q1 = project(g1, groups, groups_n)
    q2 = project(g2, groups, groups_n)
    cross_sum += float((q1 * q2).sum())
    energy_sum += float(0.5 * (q1.square().sum() + q2.square().sum()))

cross_mean = cross_sum / trials
energy_mean = energy_sum / trials
assert abs(cross_mean / float(signal_energy) - 1.0) < 0.06
assert 0.90 < cross_mean / energy_mean < 0.98
print(f"cross-microbatch estimator checks passed: numerator/signal={cross_mean/float(signal_energy):.4f}, rho={cross_mean/energy_mean:.4f}")
