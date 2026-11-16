"""SGD cluster-sharing identity and function-preserving reassignment."""
import torch
import torch.nn.functional as F

torch.manual_seed(41)
torch.set_num_threads(1)
V, K, d, B = 12, 3, 5, 7
groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
A = F.one_hot(groups, K).float()
U = torch.randn(V, d).requires_grad_()
C = torch.randn(K, d).requires_grad_()
h = torch.randn(B, d)
y = torch.randint(V, (B,))
R = U + A @ C
loss = F.cross_entropy(h @ R.T, y)
loss.backward()
p = (h @ R.detach().T).softmax(-1)
G = ((p - F.one_hot(y, V).float()).T @ h) / B
assert torch.allclose(U.grad, G, atol=1e-6)
assert torch.allclose(C.grad, A.T @ G, atol=1e-6)
eta_u, eta_c = 0.01, 0.002
R_after = U.detach() - eta_u * U.grad + A @ (C.detach() - eta_c * C.grad)
R_predict = R.detach() - (eta_u * torch.eye(V) + eta_c * A @ A.T) @ G
assert torch.allclose(R_after, R_predict, atol=2e-6)
next_groups = torch.tensor([2, 0, 1, 0, 1, 2, 0, 2, 0, 1, 1, 2])
A_next = F.one_hot(next_groups, K).float()
C_next = torch.stack([R_after[next_groups == k].mean(0) for k in range(K)])
U_next = R_after - A_next @ C_next
error = (U_next + A_next @ C_next - R_after).abs().max().item()
assert error < 1e-6
print(f"SGD exact coupled-gradient identity: PASS; regrouping max row difference={error:.3g}")
