"""Exact attention written from the assignment equations, without library SDPA.

The torch implementation mirrors Algorithm 1; the CUDA forward is in
triton_attention.py. Both save O, Q, K, V and one row log-sum-exp, never P.
"""

import torch


def attention_reference(q, k, v, is_causal=False):
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] >= torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    return scores.softmax(-1) @ v


def recompute_backward(q, k, v, o, do, l, is_causal):
    """Equations 13–19. This mandatory version can materialize quadratic tiles."""
    dtype = q.dtype
    q, k, v, o, do = (t.float() for t in (q, k, v, o, do))
    s = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] >= torch.arange(k.shape[-2], device=q.device)[None, :]
        s = s.masked_fill(~mask, -1e6)
    p = (s - l.unsqueeze(-1)).exp()
    d = (o * do).sum(-1, keepdim=True)
    dp = do @ v.transpose(-1, -2)
    ds = p * (dp - d)
    if is_causal:
        ds = ds.masked_fill(~mask, 0)
    return ((ds @ k * q.shape[-1] ** -0.5).to(dtype),
            (ds.transpose(-1, -2) @ q * q.shape[-1] ** -0.5).to(dtype),
            (p.transpose(-1, -2) @ do).to(dtype))


_compiled_backward = None


def run_backward(q, k, v, o, do, l, causal):
    global _compiled_backward
    if q.is_cuda:
        if _compiled_backward is None:
            _compiled_backward = torch.compile(recompute_backward)
        return _compiled_backward(q, k, v, o, do, l, causal)
    return recompute_backward(q, k, v, o, do, l, causal)


class FlashAttentionPyTorch(torch.autograd.Function):
    query_tile = 32
    key_tile = 64

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        nq, d = q.shape[-2:]
        nk = k.shape[-2]
        if q.shape[:-2] != k.shape[:-2] or k.shape != v.shape or k.shape[-1] != d:
            raise ValueError("Q/K/V batch and embedding dimensions must match")
        o = torch.empty_like(q)
        l = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        for start in range(0, nq, FlashAttentionPyTorch.query_tile):
            qi = q[..., start:start + FlashAttentionPyTorch.query_tile, :].float()
            m = torch.full(qi.shape[:-1], -float('inf'), device=q.device)
            den = torch.zeros_like(m)
            acc = torch.zeros_like(qi)
            for col in range(0, nk, FlashAttentionPyTorch.key_tile):
                kj = k[..., col:col + FlashAttentionPyTorch.key_tile, :].float()
                vj = v[..., col:col + FlashAttentionPyTorch.key_tile, :].float()
                s = qi @ kj.transpose(-1, -2) * d ** -0.5
                if is_causal:
                    keep = torch.arange(start, start + qi.shape[-2], device=q.device)[:, None] >= torch.arange(col, col + kj.shape[-2], device=q.device)[None, :]
                    s = s.masked_fill(~keep, -1e6)
                new_m = torch.maximum(m, s.amax(-1))
                alpha = (m - new_m).exp()
                p = (s - new_m.unsqueeze(-1)).exp()
                den = den * alpha + p.sum(-1)
                acc = acc * alpha.unsqueeze(-1) + p @ vj
                m = new_m
            o[..., start:start + qi.shape[-2], :] = (acc / den.unsqueeze(-1)).to(q.dtype)
            l[..., start:start + qi.shape[-2]] = m + den.log()
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        return (*run_backward(q, k, v, o, do, l, ctx.is_causal), None)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        from .triton_attention import forward
        o, l = forward(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        return (*run_backward(q, k, v, o, do, l, ctx.is_causal), None)


class FlashAttentionTritonFull(FlashAttentionTriton):
    """Optional two-pass Triton backward (Algorithm 2), for long sequences."""
    @staticmethod
    def backward(ctx, do):
        from .triton_attention import backward
        q, k, v, o, l = ctx.saved_tensors
        return (*backward(q, k, v, o, do, l, ctx.is_causal), None)


class FlashAttentionTritonTuned(torch.autograd.Function):
    """Separate larger-tile candidate; baseline benchmark classes stay unchanged."""
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        from .triton_attention import forward
        o, l = forward(q, k, v, is_causal, query_tile=64, key_tile=128)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        from .triton_attention import backward
        q, k, v, o, l = ctx.saved_tensors
        return (*backward(q, k, v, o, do, l, ctx.is_causal, tile=64), None)
