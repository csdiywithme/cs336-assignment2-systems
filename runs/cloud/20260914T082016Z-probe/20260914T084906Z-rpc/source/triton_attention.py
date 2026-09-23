"""Original tiled attention kernels following handout Algorithms 1 and 2."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K, V, O, L, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
             BQ: tl.constexpr, BK: tl.constexpr, is_causal: tl.constexpr):
    i, b = tl.program_id(0), tl.program_id(1)
    qp = tl.make_block_ptr(Q + b * NQ * D, (NQ, D), (D, 1), (i * BQ, 0), (BQ, D), (1, 0))
    kp = tl.make_block_ptr(K + b * NK * D, (NK, D), (D, 1), (0, 0), (BK, D), (1, 0))
    vp = tl.make_block_ptr(V + b * NK * D, (NK, D), (D, 1), (0, 0), (BK, D), (1, 0))
    q = tl.load(qp, boundary_check=(0,), padding_option="zero")
    rows = i * BQ + tl.arange(0, BQ)
    cols = tl.arange(0, BK)
    m = tl.full((BQ,), -float("inf"), tl.float32)
    den = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)
    end = NK
    if is_causal:
        end = tl.minimum(NK, (i + 1) * BQ)
    for start in range(0, end, BK):
        k = tl.load(kp, boundary_check=(0,), padding_option="zero")
        v = tl.load(vp, boundary_check=(0,), padding_option="zero")
        s = tl.dot(q, tl.trans(k), input_precision="tf32x3") * (D ** -0.5)
        s = tl.where((start + cols)[None, :] < NK, s, -float("inf"))
        if is_causal:
            s = tl.where(rows[:, None] >= (start + cols)[None, :], s, -1e6)
        new_m = tl.maximum(m, tl.max(s, 1))
        p = tl.exp(s - new_m[:, None])
        alpha = tl.exp(m - new_m)
        den = den * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc, input_precision="tf32x3")
        m = new_m
        kp = tl.advance(kp, (BK, 0))
        vp = tl.advance(vp, (BK, 0))
    op = tl.make_block_ptr(O + b * NQ * D, (NQ, D), (D, 1), (i * BQ, 0), (BQ, D), (1, 0))
    tl.store(op, (acc / den[:, None]).to(O.dtype.element_ty), boundary_check=(0,))
    tl.store(L + b * NQ + rows, m + tl.log(den), rows < NQ)


@triton.jit
def _delta(O, DO, DELTA, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    o = tl.load(O + r[:, None] * D + d[None, :], r[:, None] < N, other=0).to(tl.float32)
    do = tl.load(DO + r[:, None] * D + d[None, :], r[:, None] < N, other=0).to(tl.float32)
    tl.store(DELTA + r, tl.sum(o * do, 1), r < N)


@triton.jit
def _backward_q(Q, K, V, DO, L, DELTA, DQ, NQ: tl.constexpr, NK: tl.constexpr,
                D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr):
    i, b = tl.program_id(0), tl.program_id(1)
    qr = i * BQ + tl.arange(0, BQ)
    kd = tl.arange(0, BK)
    dd = tl.arange(0, D)
    q = tl.load(Q + b * NQ * D + qr[:, None] * D + dd[None, :], qr[:, None] < NQ, 0)
    do = tl.load(DO + b * NQ * D + qr[:, None] * D + dd[None, :], qr[:, None] < NQ, 0)
    l = tl.load(L + b * NQ + qr, qr < NQ, other=0)
    delta = tl.load(DELTA + b * NQ + qr, qr < NQ, other=0)
    dq = tl.zeros((BQ, D), tl.float32)
    end = NK
    if CAUSAL:
        end = tl.minimum(NK, (i + 1) * BQ)
    for start in range(0, end, BK):
        kr = start + kd
        k = tl.load(K + b * NK * D + kr[:, None] * D + dd[None, :], kr[:, None] < NK, 0)
        v = tl.load(V + b * NK * D + kr[:, None] * D + dd[None, :], kr[:, None] < NK, 0)
        s = tl.dot(q, tl.trans(k), input_precision="tf32x3") * D ** -0.5
        valid = (qr[:, None] < NQ) & (kr[None, :] < NK)
        if CAUSAL:
            valid = valid & (qr[:, None] >= kr[None, :])
        p = tl.where(valid, tl.exp(s - l[:, None]), 0)
        dp = tl.dot(do, tl.trans(v), input_precision="tf32x3")
        ds = p * (dp - delta[:, None]) * D ** -0.5
        dq = tl.dot(ds.to(k.dtype), k, acc=dq, input_precision="tf32x3")
    tl.store(DQ + b * NQ * D + qr[:, None] * D + dd[None, :], dq.to(DQ.dtype.element_ty), qr[:, None] < NQ)


@triton.jit
def _backward_kv(Q, K, V, DO, L, DELTA, DK, DV, NQ: tl.constexpr, NK: tl.constexpr,
                 D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr):
    j, b = tl.program_id(0), tl.program_id(1)
    kr = j * BK + tl.arange(0, BK)
    qq = tl.arange(0, BQ)
    dd = tl.arange(0, D)
    k = tl.load(K + b * NK * D + kr[:, None] * D + dd[None, :], kr[:, None] < NK, 0)
    v = tl.load(V + b * NK * D + kr[:, None] * D + dd[None, :], kr[:, None] < NK, 0)
    dk = tl.zeros((BK, D), tl.float32)
    dv = tl.zeros((BK, D), tl.float32)
    begin = 0
    if CAUSAL:
        begin = j * BK // BQ * BQ
    for start in range(begin, NQ, BQ):
        qr = start + qq
        q = tl.load(Q + b * NQ * D + qr[:, None] * D + dd[None, :], qr[:, None] < NQ, 0)
        do = tl.load(DO + b * NQ * D + qr[:, None] * D + dd[None, :], qr[:, None] < NQ, 0)
        l = tl.load(L + b * NQ + qr, qr < NQ, other=0)
        delta = tl.load(DELTA + b * NQ + qr, qr < NQ, other=0)
        s = tl.dot(q, tl.trans(k), input_precision="tf32x3") * D ** -0.5
        valid = (qr[:, None] < NQ) & (kr[None, :] < NK)
        if CAUSAL:
            valid = valid & (qr[:, None] >= kr[None, :])
        p = tl.where(valid, tl.exp(s - l[:, None]), 0)
        dv = tl.dot(tl.trans(p.to(do.dtype)), do, acc=dv, input_precision="tf32x3")
        dp = tl.dot(do, tl.trans(v), input_precision="tf32x3")
        ds = p * (dp - delta[:, None]) * D ** -0.5
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk, input_precision="tf32x3")
    tl.store(DK + b * NK * D + kr[:, None] * D + dd[None, :], dk.to(DK.dtype.element_ty), kr[:, None] < NK)
    tl.store(DV + b * NK * D + kr[:, None] * D + dd[None, :], dv.to(DV.dtype.element_ty), kr[:, None] < NK)


def forward(q, k, v, causal=False):
    if not q.is_cuda:
        raise ValueError("Triton attention requires CUDA")
    if q.shape[-1] != k.shape[-1] or k.shape != v.shape or q.shape[:-2] != k.shape[:-2]:
        raise ValueError("Incompatible Q/K/V shapes")
    d = q.shape[-1]
    if d < 16 or d & (d - 1):
        raise ValueError("head dimension must be a power of two >=16")
    batch = math.prod(q.shape[:-2])
    nq, nk = q.shape[-2], k.shape[-2]
    q, k, v = (x.contiguous() for x in (q, k, v))
    o = torch.empty_like(q)
    l = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    _forward[(triton.cdiv(nq, 32), batch)](q, k, v, o, l, nq, nk, d, 32, 64, causal, num_warps=4)
    return o, l


def backward(q, k, v, o, do, l, causal):
    q, k, v, o, do = (x.contiguous() for x in (q, k, v, o, do))
    batch = math.prod(q.shape[:-2])
    nq, nk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    delta = torch.empty_like(l)
    _delta[(triton.cdiv(batch * nq, 32),)](o, do, delta, batch * nq, d, 32)
    _backward_kv[(triton.cdiv(nk, 32), batch)](q, k, v, do, l, delta, dk, dv, nq, nk, d, 32, 32, causal, num_warps=4)
    _backward_q[(triton.cdiv(nq, 32), batch)](q, k, v, do, l, delta, dq, nq, nk, d, 32, 64, causal, num_warps=4)
    return dq, dk, dv
