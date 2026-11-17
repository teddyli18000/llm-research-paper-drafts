"""Analytic checks for rho, cosine, and stable-gradient-capture estimators."""
import torch


def project(g, groups, k):
    sums = torch.zeros(k, g.shape[1], dtype=g.dtype).index_add_(0, groups, g)
    counts = torch.bincount(groups, minlength=k).to(g.dtype)
    return sums[groups] / counts[groups, None]


def rho(a, b):
    return (a * b).sum() / (0.5 * (a.square().sum() + b.square().sum())).clamp_min(1e-30)


def cosine(a, b):
    return (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30)


def eta(a, b, g1, g2):
    return (a * b).sum() / (0.5 * (g1.square().sum() + g2.square().sum())).clamp_min(1e-30)


torch.manual_seed(2601)
vocab, width, groups_n = 128, 16, 8
groups = torch.arange(vocab) % groups_n
p = torch.randn(vocab, width)
projected = project(p, groups, groups_n)
assert torch.allclose(project(projected, groups, groups_n), projected)
assert abs(float(rho(p, p)) - 1.0) < 1e-6
assert abs(float(cosine(p, p)) - 1.0) < 1e-6
assert abs(float(eta(projected, projected, p, p)) - float(projected.square().sum() / p.square().sum())) < 1e-6
assert -1.000001 <= float(eta(projected, projected, p, p)) <= 1.000001
assert abs(float(rho(p, -p)) + 1.0) < 1e-6
assert abs(float(cosine(p, -p)) + 1.0) < 1e-6

mu = torch.randn(groups_n, width)[groups]
signal_energy = project(mu, groups, groups_n).square().sum()
cross_sum = 0.0
projected_energy_sum = 0.0
total_energy_sum = 0.0
trials = 2000
for _ in range(trials):
    g1 = mu + torch.randn(vocab, width)
    g2 = mu + torch.randn(vocab, width)
    q1 = project(g1, groups, groups_n)
    q2 = project(g2, groups, groups_n)
    cross_sum += float((q1 * q2).sum())
    projected_energy_sum += float(0.5 * (q1.square().sum() + q2.square().sum()))
    total_energy_sum += float(0.5 * (g1.square().sum() + g2.square().sum()))

cross_mean = cross_sum / trials
projected_energy_mean = projected_energy_sum / trials
total_energy_mean = total_energy_sum / trials
rho_mean = cross_mean / projected_energy_mean
eta_mean = cross_mean / total_energy_mean
capture_fraction = projected_energy_mean / total_energy_mean
assert abs(cross_mean / float(signal_energy) - 1.0) < 0.06
assert 0.90 < rho_mean < 0.98
assert 0.45 < eta_mean < 0.55
assert abs(eta_mean - rho_mean * capture_fraction) < 1e-9
print(f"stable-capture checks passed: rho={rho_mean:.4f}, capture={capture_fraction:.4f}, eta={eta_mean:.4f}")
