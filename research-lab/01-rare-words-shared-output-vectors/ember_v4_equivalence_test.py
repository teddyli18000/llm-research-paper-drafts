"""Official pinned Ember vs Oct-2026 v4 Algorithm 1 on dense full-softmax heads."""
import torch
from ember import ember_update

torch.set_num_threads(1)


@torch.no_grad()
def v4_step(param, grad, state, lr, beta2, eps):
    if not state:
        state["r"] = torch.zeros(param.size(0))
        state["c"] = torch.zeros(param.size(1))
        state["t"] = 0
    state["t"] += 1
    active = max(int(grad.ne(0).any(dim=1).sum()), 1)
    g2 = grad.square()
    state["r"].mul_(beta2).add_(g2.mean(dim=1), alpha=1-beta2)
    state["c"].mul_(beta2).add_(g2.sum(dim=0)/active, alpha=1-beta2)
    bc = 1-beta2**state["t"]
    rhat = state["r"]/bc
    chat = state["c"]/bc
    shat = torch.sqrt((rhat.mean()*chat.mean()).clamp_min(1e-30))
    vhat = rhat[:,None]*chat[None,:]/shat
    param.addcdiv_(grad, vhat.sqrt()+eps, value=-lr)


for beta2 in (0.95, 0.999):
    for seed in (0, 11, 109):
        torch.manual_seed(seed)
        official = torch.randn(128, 32)
        v4 = official.clone()
        state_official = {}
        state_v4 = {}
        for t in range(1, 16):
            # A full-softmax head has a dense gradient in every vocabulary row.
            grad = torch.randn_like(official)*0.0001
            assert grad.ne(0).any(dim=1).all()
            ember_update(official, grad, state_official, lr=0.05, beta2=beta2, eps=1e-8, wd=0.0)
            v4_step(v4, grad, state_v4, lr=0.05, beta2=beta2, eps=1e-8)
            torch.testing.assert_close(official, v4, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(state_official["r"], state_v4["r"], rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(state_official["c"], state_v4["c"], rtol=1e-5, atol=1e-7)
        print(f"EMBER_V4_EQUIVALENCE beta2={beta2} seed={seed} steps=15 PASS", flush=True)

# Sparsity distinguishes v4's active-row averaging from the pinned repository's
# all-row variant: this is expected for standalone input embedding tables.
torch.manual_seed(2026)
sparse = torch.zeros(128, 32)
sparse[:16] = torch.randn(16, 32)
w1 = torch.randn(128,32)
w2 = w1.clone()
st1, st2 = {}, {}
ember_update(w1, sparse, st1, lr=0.05, beta2=.95, eps=1e-8, wd=0.0)
v4_step(w2, sparse, st2, lr=0.05, beta2=.95, eps=1e-8)
assert not torch.allclose(w1, w2, rtol=1e-5, atol=1e-5)
print("EMBER_V4_SPARSE_VARIANT_DIFF expected=True PASS", flush=True)
