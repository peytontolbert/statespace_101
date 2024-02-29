import triton
import triton.language as tl
import torch
import math
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("whitegrid", {'axes.grid' : False})
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

def check(*inputs, prec=1e-4):
    """
    Checks if pairs of tensors are approximately equal up to a precision.
    
    Args:
    *inputs: A list of tensors to compare in pairs. Each pair is checked for closeness.
    prec: The precision tolerance for comparisons.
    
    Raises:
    AssertionError if any pair of tensors are not close up to the specified precision.
    
    Outputs:
    Prints a verification message if all pairs are close.
    """
    for i, (a, b) in enumerate(zip(inputs[::2], inputs[1::2])):
        if isinstance(b, list):
            b = torch.tensor(b)
        c = torch.allclose(a.cpu(), b.cpu(), prec)
        c1 = torch.isclose(a.cpu(), b.cpu(), prec)
        assert c, f"{i}\n{a}\n{b}\n{c1}"
    print("check")
    
def cumsum(x):
    """
    Computes the cumulative sum of a given list 'x'.
    
    Inputs:
    - x: A list or 1D tensor of numerical values.
    
    Outputs:
    - h: The final sum of the list 'x'.
    - y: A list containing the cumulative sum of 'x'.
    """
    y = []  # Initialize the list to store cumulative sums.
    h = 0  # Initialize the cumulative sum variable.
    for k in range(len(x)):  # Iterate over the elements of 'x'.
        h = h + x[k]  # Update the cumulative sum.
        y.append(h)  # Append the current cumulative sum to the list.
    return h, y  # Return the final sum and the list of cumulative sums.


@triton.jit
def plus_fn(a, b):
    # This is a helper function where a and b are tensors.
    return a + b

@triton.jit
def cumsum1_tt(X, Y, H, K: tl.constexpr):
    # This is the base triton function. Capital letters are pointers to memory.

    # Create a tensor from 0 to K - 1
    Ks = tl.arange(0, K)

    # Load in a sequence of K x's (blue)
    x = tl.load(X + Ks)

    # Compute h (green) and y (yellow) on axis 0.
    hs = tl.associative_scan(x, 0, plus_fn)
    y = hs

    # Write out K y's
    tl.store(Y + Ks, y)

    # Write out only the last h to memory.
    tl.store(H + Ks * 0, hs, mask=Ks == (K-1))

@triton.jit
def cumsum_tt(X, H_0, Y, H, K: tl.constexpr):
    # Which block an I?
    pid = tl.program_id(0)

    # How far into the sequence am I?
    kid = K * pid
    Ks = tl.arange(0, K)

    # Load in K x's per block and 1 starting h
    x = tl.load(X + Ks + kid)

    # Load the first value as H_0 and the rest 0
    h_0 = tl.load(H_0 + Ks * 0 + pid, Ks == 0, 0)

    # Allow for a starting value.
    x = plus_fn(h_0, x)

    # Compute scan
    hs = tl.associative_scan(x, 0, plus_fn)
    y = hs

    # Write out K y's per block and 1 h
    tl.store(Y + Ks + kid, y)

    # Write out only the last value to H
    tl.store(H + Ks * 0 + pid, hs, mask=Ks == (K-1))


def cumsum_block(x, y, K):
    seqlen = y.shape[0]
    BLOCKS = seqlen // K
    h = zeros(2, BLOCKS)
    cumsum_tt[(BLOCKS,)](x, h[0], y, h[0], K=K)
    h[1, 1:] = h[0].cumsum(0)[:-1]
    cumsum_tt[(BLOCKS,)](x, h[1], y, h[1], K=K)

alpha = 0.9
def ema(x, alpha):
    y = []
    h = 0
    for k in range(len(x)):
        h = alpha * h + (1-alpha) * x[k]
        y.append(1 * h)
    return h, y

def ssm_scan(x, a, b, c):
    y = []
    h = 0
    for k in range(len(x)):
        h = h * a + b * x[k]
        y.append(c * h)
    return h, y

def op(a, b):
    return (a[0] * b[0], b[0] * a[1] + b[1])

def ssm_associative(x, a, b, c):
    y = []
    h = (alpha, 0)
    for k in range(len(x)):
        h_new = (a, b * x[k])
        h = op(h, h_new)
        y.append(c * h[1])
    return h, torch.stack(y)






# implementation 
@triton.jit
def first_order_op(fl, xl, fr, xr):
    f = fr * fl
    x = fr * xl + xr
    return f, x
@triton.jit
def ssm_load(Ks, A, B, C):
    "Helper for loading"
    a = tl.load(A + Ks)
    b = tl.load(B + Ks)
    c = tl.load(C + Ks)
    return a, b, c

@triton.jit
def simple_ssm_tt(X, A, B, C, Y, K: tl.constexpr):
    Ks = tl.arange(0, K)

    # Allow for a batch dimension (for Part 4)
    bid = tl.program_id(0)
    kid = bid * K
    x = tl.load(X + Ks + kid)
    a, b, c = ssm_load(Ks + kid, A, B, C)

    # Compute
    h1, h2 = tl.associative_scan((a, b*x), 0, first_order_op)
    y = c * h2

    # Save
    tl.store(Y + Ks + kid, y)

h = torch.zeros(2, BLOCKS ).float().cuda()
a, b, c = ones(SEQLEN) * alpha, ones(SEQLEN) - alpha, ones(SEQLEN)
simple_ssm_tt[(1,)](x, a, b, c, y, K=K)
h_, y_ = ema(x[:K].tolist(), alpha)
check(y[:K], y_)

@triton.jit
def ssm_scan(h1, h2, h2_0, reversed:tl.constexpr=0, dim:tl.constexpr=0):
    # Optional flip direction (for Part 3)
    Ks = tl.arange(0, h2.shape[dim])
    # Apply initial
    n1, n2 = first_order_op(tl.zeros_like(h1)+1.0, h2_0, h1, h2)

    # Scan
    h1, h2 = tl.associative_scan((n1, n2), dim, first_order_op, reverse=reversed)
    return h1, h2

@triton.jit
def ema_tt(X, A, B, C, H_0, Y, H, K: tl.constexpr):
    pid = tl.program_id(0)
    nH = tl.num_programs(0)
    Ks = tl.arange(0, K)
    kid = pid * K
    a, b, c = ssm_load(Ks + kid, A, B, C)
    x = tl.load(X + Ks + kid)
    h_span = Ks*0 + pid
    h2_0 = tl.load(H_0 + nH + h_span, Ks==0, 0)

    # Compute
    h1, h2 = ssm_scan(a, b * x, h2_0, 0)

    # Save
    tl.store(Y + Ks + kid, h2)

    # Write out two part hidden state.
    tl.store(H + 0 * nH + h_span, h1, Ks == (K-1))
    tl.store(H + 1 * nH + h_span, h2, Ks == (K-1))

h = torch.zeros(2, 2, BLOCKS).float().cuda()
_ = torch.zeros(K * BLOCKS).cuda()
o = ones(BLOCKS)

ema_tt[(BLOCKS,)](x, a, b, c, h[0], y, h[0], K=K)
simple_ssm_tt[(1,)](h[0, 1], h[0, 0], o, o, h[1, 1], K=BLOCKS)
ema_tt[(BLOCKS,)](x, a, b, c, torch.roll(h[1], 1), y, h[1], K=K)

h_, y_ = ema(x.tolist(), alpha)
check(y, y_)

a, b, c = rand((SEQLEN,)), rand((SEQLEN,)), rand((SEQLEN,))
def ssm_torch(x, a, b, c):
    y = []
    h = 0
    for k in range(len(x)):
        h = a[k] * h + b[k] * x[k]
        y.append(c[k] * h)
    return h, torch.stack(y)

def L(x, a, b, c):
    return ssm_torch(x, a, b, c)[1].sum()

x_ = x.clone()
h, y_ = ssm_torch(x_, a, b, c)

g = torch.func.grad(L, tuple(range(4)))
dx_, da_, db_, dc_ = g(x_, a, b, c)

plt.bar(range(SEQLEN), dx_.cpu())

dy, dx = ones(SEQLEN), ones(SEQLEN)
da, db, dc = [torch.zeros(K*BLOCKS).float().cuda() for _ in range(3)]
_, _ign = torch.zeros(K * BLOCKS).cuda(), torch.zeros(K * BLOCKS).cuda()

simple_ssm_tt[(1,)](dy.flip(0), a.flip(0).roll(1), c.flip(0), b.flip(0), dx, K=SEQLEN)
dx = dx.flip(0)
check(dx, dx_)
plt.bar(range(SEQLEN), dx.cpu())


@triton.jit
def rol(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur

@triton.jit
def roll(y, dim, rev=0):
    _, rh2, _ = tl.associative_scan((1 + 0*y, 0.0*y, y), dim, rol, reverse=rev)
    return rh2

@triton.jit
def ssm_store(Ks, dA, da, dB, db, dC, dc):
    "Helper"
    tl.store(dA + Ks, da)
    tl.store(dB + Ks, db)
    tl.store(dC + Ks, dc)




@triton.jit
def ssm1_tt(X, dX, A, dA, B, dB, C, dC, Y, dY, K: tl.constexpr):
    Ks = tl.arange(0, K)
    a, b, c = ssm_load(Ks, A, B, C)
    x = tl.load(X + Ks)
    dy = tl.load(dY + Ks)
    id2 = tl.zeros_like(a) # 0.0

    # Compute Forward (same as before)
    h1, h2 = ssm_scan(a, b * x, id2)
    y = c * h2
    tl.store(Y + Ks, y)
    a_shift = tl.load(A + Ks + 1, Ks + 1 < K, 0)

    # Compute Backward (now reversed)
    h1, dh = ssm_scan(a_shift, c * dy, id2, reversed=1)
    rh2 = roll(h2, 0)

    # Save
    tl.store(dX + Ks, b * dh)
    ssm_store(Ks, dA, dh*rh2, dB, dh*x, dC, h2 * dy)

dx, da, db, dc = [torch.zeros(SEQLEN).float().cuda() for _ in range(4)]
dy = torch.ones(SEQLEN).float().cuda()

ssm1_tt[(1,)](x, dx, a, da, b, db, c, dc, y, dy, K=SEQLEN)
check(da, da_, dc, dc_, dx, dx_, db, db_)
plt.bar(range(SEQLEN), da.cpu())

# Group together all the inputs and all the grads.
@triton.jit
def ssm_tt(X, dX, A, dA, B, dB, C, dC, H_0, dH_0, H, dH, Y, dY,
           back: tl.constexpr,
           K: tl.constexpr):
    pid = tl.program_id(0)
    nH = tl.num_programs(0)

    Ks = tl.arange(0, K)
    kid = pid * K

    # Load
    x = tl.load(X + Ks + kid)
    a, b, c = ssm_load(Ks + kid, A, B, C)
    h2_0 = tl.load(H_0 + nH + Ks*0 + pid, Ks==0, 0)

    # # Compute Forward (Move L-to-R)
    h1, h2 = ssm_scan(a, b * x, h2_0)
    y = c * h2

    tl.store(Y + Ks + kid, y)
    tl.store(H + 0*nH + Ks * 0 + pid, h1, Ks == K-1)
    tl.store(H + 1*nH + Ks * 0 + pid, h2, Ks == K-1)
    if not back: return

    # Compute Backward (Move R-to-L)
    dy = tl.load(dY + Ks + kid)
    a_shift = tl.load(A + Ks + kid + 1)

    dh_0 = tl.load(dH_0 + nH + pid) * (Ks==K-1)
    dh1, dh = ssm_scan(a_shift, c * dy, dh_0, reversed=1)
    rh2 = roll(h2, 0) + h2_0


    # Save
    tl.store(dX + Ks + kid, b * dh)
    ssm_store(Ks + kid, dA, dh*rh2, dB, dh*x, dC, h2 * dy)
    tl.store(dH + 0*nH + Ks * 0 + pid, dh1, Ks == 0)
    tl.store(dH + 1*nH + Ks * 0 + pid, dh,  Ks == 0)

h, dh = (zeros(2, 2, BLOCKS) for _ in range(2))
dx = zeros(SEQLEN)

def run(h, dh):
    ssm_tt[(BLOCKS,)](x, dx, a, da, b, db, c, dc, h, dh, h, dh, y, dy,
                      back=1, K=K)

def reduce(v, rev, batch = 1):
    if rev:
        v[0, :] = v[0].flip(-1)
    o = torch.ones_like(v[0, 0])
    simple_ssm_tt[(batch,)](v[0, 1], v[0, 0], o, o, v[1, 1], K=v.shape[-1])
    v[..., -1] = 0.0
    v[:] = torch.roll(v, 1)
    if rev:
        v[1, :] = v[1].flip(-1)

run(h[0], dh[0])
reduce(h, False)
reduce(dh, True)
run(h[1], dh[1])

dx_, da_, b_, dc_ = g(x, a, b, c)

#check(dx, dx_, dc, dc_, db, db_, da, da_)

N = 4
def ssm_multiscan(x, a, b, c):
    y = []
    h = zeros(N)
    for k in range(len(x)):
        h = h * a[:, k] + b[:, k] * x[k]
        y.append((c[:, k] * h).sum(0))
    return h, torch.stack(y)

alpha = (((arange(N) + 1) / 8)[:, None]).expand((N, SEQLEN)).clone()
a, b, c = alpha, (1-alpha), ones(N, SEQLEN)
h_, y_ = ssm_multiscan(x, a, b, c)
plt.bar(range(SEQLEN), y_.cpu())

@triton.jit
def select(X, mask, dim=-1):
    return tl.sum(X * mask, dim, 1)

@triton.jit
def multiema_tt(X, A, B, C, H_0, Y, H, K: tl.constexpr,
                L: tl.constexpr, N: tl.constexpr):
    # L is the total length of the sequence.
    pid = tl.program_id(0)
    nH = tl.num_programs(0) # Number of blocks.
    Ks = tl.arange(0, K)[None, :]
    Ns = tl.arange(0, N)[:, None] # N x 1 - For each hidden.
    kid = pid * K
    h_span = Ns*nH + pid

    a, b, c = ssm_load(Ns * L + Ks + kid, A, B, C) # N x K
    x = tl.load(X + Ks + kid) # K
    h2_0 = tl.load(H_0 + nH*N + h_span) * (Ks==0)

    # Compute forward for all hidden.
    h1, h2 = ssm_scan(a, b * x, h2_0, dim=1)
    y = tl.sum(c * h2, 0)

    # Save
    tl.store(Y + Ks + kid, y[None, :])
    tl.store(H + 0 * nH*N + h_span, select(h1, Ks == (K-1)))
    tl.store(H + 1 * nH*N + h_span, select(h2, Ks == (K-1)))
    

N = 4
h = zeros(2, 2, N, BLOCKS)
o = ones(N, BLOCKS)
multiema_tt[(BLOCKS,)](x, a, b, c, h[0], y, h[0], K=K, L=x.shape[0], N=N)
simple_ssm_tt[(N,)](h[0, 1], h[0, 0], o, o, h[1, 1], K=BLOCKS)
h[..., -1] = 0
multiema_tt[(BLOCKS,)](x, a, b, c, torch.roll(h[1], 1, -1), y, h[1], K=K, L=x.shape[0], N=N)
check(y, y_)

# MAMBA
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
mamba1_tt[(1,)](x, dx, a, da, b, db, c, dc, delta, ddelta, y, dy, K=SEQLEN)
plt.bar(range(SEQLEN), y.cpu())