import triton
import triton.language as tl
import torch
import math
from matplotlib import pyplot as plt
import selective_scan_cuda
from utils import check, roll, ssm_load, ssm_scan, reduce, ssm_store
# Utility functions for creating tensors on the GPU.
ones = lambda *size: torch.ones(*size).float().cuda()  # Creates a tensor of ones.
zeros = lambda *size: torch.zeros(*size).float().cuda()  # Creates a tensor of zeros.
arange = lambda n: torch.arange(n).float().cuda()  # Creates a range tensor from 0 to n-1.
rand = lambda size: torch.rand(*size).abs().float().cuda()  # Creates a tensor of random numbers.
K = 16  # Length of the sequence in each block.
BLOCKS = 8  # Number of blocks.
SEQLEN = K * BLOCKS  # Total length of the sequence to be processed.

# These tensors are prepared for the cumulative sum calculation.
x = arange(SEQLEN) # Create a sequence 'x' from 0 to SEQLEN-1
y = zeros(SEQLEN) # Zero-initialized tensor 'y' of the same length.

x_ = x.clone()
# Mamba Implementation
def discretize(a, b, delta):
    da = delta * a
    a_ = torch.exp(da)
    b_ = b * delta
    return a_, b_
ax = plt.figure().add_subplot(projection='3d')
a = torch.linspace(-1, 1, 100)[None].expand(100, 100)
delta = torch.linspace(-1, 1, 100)[:, None].expand(100, 100)
ax.plot_surface(a.cpu(), delta.cpu(), discretize(a, 1, delta)[0].cpu())
ax.set_xlabel('$A$')
ax.set_ylabel('$\Delta$')

x = arange(SEQLEN)
a, b, c, delta = [ones(SEQLEN) for _ in range(4)]
delta[:] = 0.01
def simple_mamba_torch(x, a, b, c, delta):
    y = []
    h = 0
    a_, b_ = discretize(a, b, delta)
    for k in range(len(x)):
        h = a_[k] * h + b_[k] * x[k]
        y.append(c[k] * h)
    return h, torch.stack(y)

def L(x, a, b, c, delta):
    return simple_mamba_torch(x, a, b, c, delta)[1].sum()

h, y_ = simple_mamba_torch(x, a, b, c, delta)
g = torch.func.grad(L, tuple(range(5)))
dx_, da_, db_, dc_, ddelta_ = g(x_, a, b, c, delta)

delt = 0.5
alpha = 0.9
discretize(torch.tensor(alpha).log() / delt, -torch.tensor(alpha).log(), torch.tensor(delt))

@triton.jit
def discretize_tt(a, b, delta):
    da = delta * a
    a_ = tl.exp(da)
    b_ = b * delta
    return a_, b_

@triton.jit
def discretize_back(a, b, d, da_, db_):
    da = d * a
    a_ = tl.exp(da)

    da_da = d * a_
    da_ddelta = a * a_

    inter = (b * (da - 1) * a_ + b) / da

    #db_da = 0
    db_db = d
    db_ddelta = b

    return da_ * da_da, db_ * db_db, da_ * da_ddelta + db_ * db_ddelta


@triton.jit
def mamba1_tt(X, dX, A, dA, B, dB, C, dC, Delta, dDelta, Y, dY, K: tl.constexpr):
    Ks = tl.arange(0, K)
    a, b, c = ssm_load(Ks, A, B, C)
    x = tl.load(X + Ks)
    dy = tl.load(dY + Ks)
    delta = tl.load(Delta + Ks)
    id2 = Ks * 0.0

    # Compute Forward
    a_, b_ = discretize_tt(a, b, delta)
    h1, h2 = ssm_scan(a_, b_ * x, id2)
    y = c * h2
    tl.store(Y + Ks, y)

    # Compute Backward
    h1, dh = ssm_scan(roll(a_, 0, 1), c * dy, id2, reversed=1)
    rh2 = roll(h2, 0)
    da_ = dh*rh2
    db_ = dh*x
    da, db, ddelta = discretize_back(a, b, delta, da_, db_)

    # Save
    tl.store(dDelta + Ks, ddelta)
    tl.store(dX + Ks, b_ * dh)
    ssm_store(Ks, dA, da, dB, db, dC, h2 * dy)

dx, da, db, dc, ddelta = [zeros(SEQLEN) for _ in range(5)]
y, dy = [ones(SEQLEN) for _ in range(2)]
mamba1_tt[(1,)](x, dx, a, da, b, db, c, dc, delta, ddelta, y, dy, K=SEQLEN)
dx_, da_, db_, dc_, ddelta_ = g(x_, a, b, c, delta)
check(y, y_, dx, dx_, da[-2*K:], da_[-2*K:], db[-2*K:], db_[-2*K:], ddelta[-2*K:], ddelta_[-2*K:])

a = (torch.zeros_like(a) + 0.9).log()
b = -(torch.zeros_like(b) + 0.9).log()
delta = torch.ones_like(delta)
result = mamba1_tt[(1,)](x, dx, a, da, b, db, c, dc, delta, ddelta, y, dy, K=SEQLEN)
print(result)



# Block Implementation
def mamba_torch(x, a, b, c, delta):
    "PyTorch Implementation"
    y = []
    h = 0
    a_, b_ = discretize(a, b, delta)
    for k in range(x.shape[-1]):
        h = a_[..., k] * h + b_[..., k] * x[..., k]
        y.append((c[..., k] * h).sum(1, keepdim=True))
    return h, torch.stack(y, -1)

@triton.jit
def mamba_tt(X, dX, A, dA, B, dB, C, dC, Delta, dDelta,
             H_0, dH_0, Y, dY, H, dH,
             back:tl.constexpr,
             step:tl.constexpr,
             L: tl.constexpr, K: tl.constexpr, D_step: tl.constexpr,
             D:tl.constexpr, N: tl.constexpr):
    # Setup
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    kid = pid * K
    nH = tl.num_programs(0)
    Ba = tl.num_programs(1)
    Ks = tl.arange(0, K)[None, None, :] # 1 x 1 x K
    Ns = tl.arange(0, N)[:, None, None] # N x 1 x 1
    Nx1xK = bid*N*L + Ns*L + (Ks + kid)



    # Load forward
    b = tl.load(B + Nx1xK)
    c = tl.load(C + Nx1xK)
    db_out = tl.zeros_like(b)
    dc_out = tl.zeros_like(c)

    Ds = tl.arange(0, D_step)[None, :, None] # 1 x D x 1

    for did in range(0, D // D_step):
        DxK = bid*D*L + Ds*L + Ks + kid
        NxDx1 = bid*N*D + Ns*D + Ds
        a = tl.load(A + NxDx1)
        NxDx1_H = bid*N*D*nH + Ns*D*nH + Ds*nH + pid
        h_off = Ba*N*D*nH

        # Load forward
        delta = tl.load(Delta + DxK)
        x = tl.load(X + DxK)
        a_, b_ = discretize_tt(a, b, delta)

        if step == 2:
            h2_0 = tl.load(H_0 + 1*h_off + NxDx1_H) * (Ks == 0)
        else:
            h2_0 = tl.zeros_like(a_)
        # Compute Forward
        h1, h2 = ssm_scan(a_, b_ * x, h2_0, dim=2)
        y = tl.sum(c * h2, 0, 1)
        if step == 1:
            tl.store(H + 0 * h_off + NxDx1_H + 0*Ks, h1, Ks==K-1)
            tl.store(H + 1 * h_off + NxDx1_H + 0*Ks, h2, Ks==K-1)
        if step == 2:
            tl.store(Y + DxK, y)

        # #Compute backward
        if back == 1:
            # Load Backward
            dy = tl.load(dY + DxK)
            dh2_0 = tl.load(dH_0 + 1*h_off + NxDx1_H) * (Ks==K-1)
            delta_shift = tl.load(Delta + DxK + 1, (Ks + kid) < L - 1, 0)
            a_s, _ = discretize_tt(a, b, delta_shift)
            dh1, dh = ssm_scan(a_s, c * dy, dh2_0, reversed=1, dim=2)
            if step == 1:
                tl.store(dH + 0*h_off + NxDx1_H + 0*Ks, dh1, Ks == 0)
                tl.store(dH + 1*h_off + NxDx1_H + 0*Ks, dh, Ks == 0)

        if back == 1 and step == 2:
            dc = tl.sum(h2 * dy, 1, 1) # N x K
            rh2 = roll(h2, 2)
            rh2 = h2_0 * (Ks == 0) + rh2 * (Ks > 0)
            da, db, ddelta = discretize_back(a, b, delta, dh * rh2, dh * x)

            # Save (sums keep_dims=1)
            tl.store(dX + DxK, tl.sum(b_ * dh, 0, 1))
            tl.store(dA + NxDx1_H, tl.sum(da, 2, 1))
            tl.store(dDelta + DxK, tl.sum(ddelta, 0, 1))
            db_out = db_out + tl.sum(db, 1, 1)
            dc_out = dc_out + dc
        Ds = Ds + D_step

    if back==1 and step==2:
        tl.store(dB + Nx1xK, db_out)
        tl.store(dC + Nx1xK, dc_out)


def create(S = 128, Ba = 2, D = 4, N = 4):
    x = rand((Ba, 1, D, S))
    a = -ones((Ba, N, D, 1))
    b = ones((Ba, N, 1, S)) * 0.1
    c = rand((Ba, N, 1, S)) * 0.1
    delta = rand((Ba, 1, D, S)) * 0.1
    return x, a, b, c, delta

def mamba(x, a, b, c, delta, K=16, D_step=2):
    Ba = x.shape[0]
    N = a.shape[1]
    D = delta.shape[2]
    SEQLEN = x.shape[-1]
    BLOCKS = SEQLEN // K
    dx, da, db, dc, ddelta = [torch.zeros_like(b) for b in [x,a,b,c,delta]]
    da = zeros(Ba, N, D, BLOCKS)
    y, dy = [ones(Ba, 1, D, SEQLEN) for _ in range(2)]
    h, dh = [zeros(2, 2, Ba, N, D, BLOCKS) for _ in range(2)]
    assert BLOCKS == SEQLEN // K
    assert D % D_step == 0
    mamba_tt[(BLOCKS, Ba)](x, dx, a, da, b, db, c, dc, delta, ddelta, h[0], dh[0], y, dy, h[0], dh[0], back=1, step=1, L=SEQLEN, K=K, D_step=D_step, D=D, N=N)
    reduce(h, False, Ba * N * D)
    reduce(dh, True, Ba * N * D)
    mamba_tt[(BLOCKS, Ba)](x, dx, a, da, b, db, c, dc, delta, ddelta, h[1], dh[1], y, dy, h[1], dh[1], back=1, step=2, L=SEQLEN, K=K, D_step=D_step, D=D, N=N)
    return y, dx, da.sum(-1, keepdim=True), db, dc, ddelta

x, a, b, c, delta = create()
y, dx, da, db, dc, ddelta = mamba(x, a, b, c, delta)
for v in [x, a, b, c, delta]:
    v.requires_grad_()
_, y_ = mamba_torch(x, a, b, c, delta)
y_.sum().backward()
check(y, y_, da, a.grad, dx, x.grad, dc, c.grad, db, b.grad, prec=1e-3)

y_from_repo = selective_scan_cuda.fwd(x.squeeze(1), delta.squeeze(1), a[0].squeeze(-1).T, b.squeeze(-2)[:, None, :, :], c.squeeze(-2)[:, None, :, :], None, None, None, False)

check(y.squeeze(1), y_from_repo[0])

y_ = mamba(*create(S = 2048, Ba = 8, D = 1024, N=4), K = 128, D_step=16)