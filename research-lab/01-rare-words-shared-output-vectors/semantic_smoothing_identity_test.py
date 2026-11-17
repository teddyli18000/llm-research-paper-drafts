"""Verify head-only class-smoothed gradient and projected-softmax identities."""
import torch
import torch.nn.functional as F

torch.manual_seed(2026)
torch.set_num_threads(1)
N, V, d, K = 9, 107, 11, 7
h = torch.randn(N, d)
t = torch.randint(V, (N,))
groups = torch.randint(K, (V,))
counts = torch.bincount(groups, minlength=K).clamp_min(1)

w = torch.randn(V, d, requires_grad=True)
z = F.linear(h, w)
p = F.softmax(z, -1)
lp = F.log_softmax(z, -1)
ce = F.cross_entropy(z, t)

# Actual smoothed CE, with unit-mass target spread over the target group.
group_lp = torch.zeros(N, K).index_add_(1, groups, lp)
soft_ce = (1 - .5) * (-lp.gather(1,t[:,None]).squeeze(1)) + .5 * (
    -group_lp.gather(1,groups[t][:,None]).squeeze(1) / counts[groups[t]]
)
actual = torch.autograd.grad(soft_ce.mean(), w, retain_graph=True)[0]
hard = torch.autograd.grad(ce, w, retain_graph=True)[0]

pooled_targets = torch.zeros(K, d).index_add_(0, groups[t], h)
analytic = hard.clone()
analytic.index_add_(0, t, .5*h/N)
analytic.sub_(.5 * pooled_targets[groups] / counts[groups,None] / N)
torch.testing.assert_close(analytic, actual, rtol=2e-5, atol=2e-6)
print("TARGET_ONLY_LABEL_SMOOTHING_HEAD_GRADIENT PASS")

# Projector formula and the explicit predicted minus target group residual.
alpha = 2.0
projected = torch.zeros(K,d).index_add_(0,groups,hard)[groups] / counts[:,None]
via_operator = hard + alpha*projected

onehot = F.one_hot(t, num_classes=V)
group_p = torch.zeros(N,K).index_add_(1,groups,p)
group_y = torch.zeros(N,K).index_add_(1,groups,onehot.float())
class_error = (group_p-group_y)[:,groups] / counts[groups]
row_error = p-onehot.float() + alpha*class_error
via_logits = row_error.T @ h / N
torch.testing.assert_close(via_operator, via_logits, rtol=2e-5, atol=2e-6)
assert abs(via_operator.sum(0).abs().max().item()) < 1e-5
print("CLASS_PROJECTED_SOFTMAX_RESIDUAL_IDENTITY PASS")
