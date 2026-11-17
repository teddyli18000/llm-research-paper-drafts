"""Exact fixed-partition SGD equivalence of I + alpha P and normalized shared centroids."""
import torch
import torch.nn.functional as F


torch.manual_seed(2601)
vocab, width, groups_n, batch, steps = 37, 11, 7, 13, 25
assignment = torch.arange(vocab) % groups_n
assignment = assignment[torch.randperm(vocab)]
A = F.one_hot(assignment, num_classes=groups_n).to(torch.float64)
D = A.T @ A
D_inv = torch.diag(1.0 / D.diag())
D_inv_sqrt = torch.diag(D.diag().rsqrt())
P = A @ D_inv @ A.T
alpha, lr = 2.0, 0.017
S = alpha**0.5 * A @ D_inv_sqrt

W_direct = torch.randn(vocab, width, dtype=torch.float64)
U = W_direct.clone()
C = torch.zeros(groups_n, width, dtype=torch.float64)

for step in range(steps):
    h = torch.randn(batch, width, dtype=torch.float64)
    target = torch.randint(vocab, (batch,))
    logits = h @ W_direct.T
    prob = logits.softmax(dim=-1)
    residual = prob
    residual[torch.arange(batch), target] -= 1
    G = residual.T @ h / batch

    direct_update = G + alpha * (P @ G)
    W_direct = W_direct - lr * direct_update

    grad_U = G
    grad_C = S.T @ G
    U = U - lr * grad_U
    C = C - lr * grad_C
    W_factored = U + S @ C

    error = (W_direct - W_factored).abs().max().item()
    assert error < 2e-12, (step, error)

print(f"NORMALIZED_CENTROID_SGD_EQUIVALENCE PASS steps={steps} max_error={error:.3e}")
