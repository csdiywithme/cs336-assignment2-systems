import triton
import triton.language as tl


@triton.jit
def _adamw(P, G, M, V, N: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
           ALPHA, DECAY: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    p, g = tl.load(P + i, mask, 0), tl.load(G + i, mask, 0)
    m, v = tl.load(M + i, mask, 0), tl.load(V + i, mask, 0)
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    p = p - DECAY * p - ALPHA * m / (tl.sqrt(v) + EPS)
    tl.store(P + i, p, mask)
    tl.store(M + i, m, mask)
    tl.store(V + i, v, mask)


def update(p, g, m, v, b1, b2, alpha, decay, eps):
    _adamw[(triton.cdiv(p.numel(), 1024),)](p, g, m, v, p.numel(), b1, b2, alpha, decay, eps, 1024)
