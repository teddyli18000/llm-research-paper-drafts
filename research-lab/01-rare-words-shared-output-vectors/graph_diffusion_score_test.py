"""Verify graph-diffusion, P-limit, and resistance score formulas and softmax gradients."""
import torch
import torch.nn.functional as F


torch.manual_seed(6831)
dtype = torch.float64
groups = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
vocab, k = len(groups), 3
A = F.one_hot(groups, num_classes=k).to(dtype)
sizes = A.sum(dim=0)
D_inv = torch.diag(1.0 / sizes)
P = A @ D_inv @ A.T

adj = torch.zeros(vocab, vocab, dtype=dtype)
for g in range(k):
    idx = torch.where(groups == g)[0]
    adj[idx[:, None], idx[None, :]] = 1
    adj[idx, idx] = 0
degree = torch.diag(adj.sum(dim=1))
L = degree - adj
L_pinv = torch.linalg.pinv(L)
tau = 0.07
H = torch.matrix_exp(-tau * L / 2)

z = torch.randn(vocab, dtype=dtype, requires_grad=True)
target = torch.tensor(4)
p = z.softmax(dim=0)
y = F.one_hot(target, num_classes=vocab).to(dtype)
r = p - y
J = torch.diag(p) - torch.outer(p, p)

p_score = r @ P @ r
diffusion_score = r @ H @ r
resistance_score = r @ L_pinv @ r

grad_p = torch.autograd.grad(p_score, z, retain_graph=True)[0]
grad_d = torch.autograd.grad(diffusion_score, z, retain_graph=True)[0]
grad_r = torch.autograd.grad(resistance_score, z)[0]

assert torch.allclose(grad_p, 2 * J @ P @ r, atol=1e-12, rtol=1e-10)
assert torch.allclose(grad_d, 2 * J @ H @ r, atol=1e-12, rtol=1e-10)
assert torch.allclose(grad_r, 2 * J @ L_pinv @ r, atol=1e-12, rtol=1e-10)

group_resid = A.T @ r
p_score_groups = (group_resid.square() / sizes).sum()
within = r - P @ r
within_sq = torch.stack([within[groups == g].square().sum() for g in range(k)])
decay = torch.exp(-tau * sizes / 2)
diffusion_groups = p_score_groups + (decay * within_sq).sum()
resistance_groups = (within_sq / sizes).sum()

assert torch.allclose(p_score, p_score_groups, atol=1e-12, rtol=1e-10)
assert torch.allclose(diffusion_score, diffusion_groups, atol=1e-12, rtol=1e-10)
assert torch.allclose(resistance_score, resistance_groups, atol=1e-12, rtol=1e-10)

print(
    "GRAPH_DIFFUSION_SCORE_IDENTITY PASS "
    f"p_score={p_score.item():.6f} "
    f"diffusion={diffusion_score.item():.6f} "
    f"resistance={resistance_score.item():.6f}"
)
