"""SGD equivalence: normalized group projector versus linear class reparameterization."""
import torch
import torch.nn.functional as F

torch.manual_seed(2026)
torch.set_num_threads(1)
V,d,N=83,13,7
assignment=torch.cat((torch.zeros(24,dtype=torch.long),
                       torch.ones(4,dtype=torch.long),
                       torch.full((55,),2,dtype=torch.long)))
P=torch.zeros(V,V)
for g in range(3):
    idx=(assignment==g).nonzero().flatten()
    P[idx[:,None],idx[None,:]]=1.0/len(idx)
eye=torch.eye(V)

for alpha in (0.0,0.5,1.0,2.0,7.0):
    B=eye+((1+alpha)**0.5-1)*P
    torch.testing.assert_close(B@B,eye+alpha*P,rtol=1e-5,atol=1e-5)
    h=torch.randn(N,d)
    y=torch.randint(V,(N,))
    U=torch.randn(V,d,requires_grad=True)
    R=(B@U).detach().requires_grad_()
    loss=F.cross_entropy(F.linear(h,R),y)
    G=torch.autograd.grad(loss,R)[0]
    counterpart=F.cross_entropy(F.linear(h,B@U),y)
    dU=torch.autograd.grad(counterpart,U)[0]
    eta=0.004
    torch.testing.assert_close(B@(U-eta*dU),R-eta*(G+alpha*P@G),rtol=1e-5,atol=2e-5)
    torch.testing.assert_close(G.sum(0),torch.zeros(d),rtol=1e-4,atol=1e-6)
    torch.testing.assert_close((P@G).sum(0),torch.zeros(d),rtol=1e-4,atol=1e-6)
    print(f"TREE_LOSS_EQ alpha={alpha} unequal_group_sizes=[24,4,55] PASS",flush=True)
