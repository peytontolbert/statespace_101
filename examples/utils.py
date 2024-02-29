import torch
import triton.language as tl
import triton
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
    y = []
    h = 0
    for k in range(len(x)):
        h = h + x[k]
        y.append(h)
    return h, y    

def ssm_scan(x, a, b, c):
    y = []
    h = 0
    for k in range(len(x)):
        h = h * a + b * x[k]
        y.append(c * h)
    return h, y

@triton.jit
def rol(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur

@triton.jit
def roll(y, dim, rev=0):
    _, rh2, _ = tl.associative_scan((1 + 0*y, 0.0*y, y), dim, rol, reverse=rev)
    return rh2

@triton.jit
def ssm_load(Ks, A, B, C):
    "Helper for loading"
    a = tl.load(A + Ks)
    b = tl.load(B + Ks)
    c = tl.load(C + Ks)
    return a, b, c


@triton.jit
def ssm_store(Ks, dA, da, dB, db, dC, dc):
    "Helper"
    tl.store(dA + Ks, da)
    tl.store(dB + Ks, db)
    tl.store(dC + Ks, dc)

@triton.jit
def first_order_op(fl, xl, fr, xr):
    f = fr * fl
    x = fr * xl + xr
    return f, x

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
    
def reduce(v, rev, batch = 1):
    if rev:
        v[0, :] = v[0].flip(-1)
    o = torch.ones_like(v[0, 0])
    simple_ssm_tt[(batch,)](v[0, 1], v[0, 0], o, o, v[1, 1], K=v.shape[-1])
    v[..., -1] = 0.0
    v[:] = torch.roll(v, 1)
    if rev:
        v[1, :] = v[1].flip(-1)
