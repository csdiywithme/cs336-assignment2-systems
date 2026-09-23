"""Assignment-owned memory-efficient path: tiled attention and chunked LM loss.

No torch SDPA, external FlashAttention, FSDP, or DDP implementations are used.
The logits-producing forward is preserved. training_loss uses an equivalent
chunked projection / cross entropy that computes dX and dW immediately.
"""
import math
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from .attention import FlashAttentionTritonFull, FlashAttentionTritonTuned


def install_flash_attention(model, tuned=False):
    from cs336_basics.model import CausalMultiHeadSelfAttention
    def forward(self, x, token_positions=None):
        b, n, _ = x.shape
        q, k, v = [projection(x).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
                   for projection in (self.q_proj, self.k_proj, self.v_proj)]
        if self.positional_encoder is not None:
            positions = token_positions.unsqueeze(-2) if token_positions is not None else None
            q, k = self.positional_encoder(q, positions), self.positional_encoder(k, positions)
        # The reference bmm is autocast after RoPE, not before it.
        if torch.is_autocast_enabled("cuda"):
            dtype = torch.get_autocast_dtype("cuda")
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        else:
            q, k = q.to(v.dtype), k.to(v.dtype)
        cls = FlashAttentionTritonTuned if tuned else FlashAttentionTritonFull
        out = cls.apply(q, k, v, True)
        return self.output_proj(out.transpose(1, 2).contiguous().view(b, n, self.d_model))
    for module in model.modules():
        if isinstance(module, CausalMultiHeadSelfAttention):
            module.forward = MethodType(forward, module)
    return model


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, chunk, compute_dtype):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        labels = targets.flatten()
        total = flat.shape[0]
        dx = torch.empty_like(flat)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        w = weight.to(compute_dtype)
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        for start in range(0, total, chunk):
            inputs = flat[start:start + chunk].to(compute_dtype)
            logits = (inputs @ w.T).float()
            rows = torch.arange(logits.shape[0], device=x.device)
            labels_part = labels[start:start + chunk]
            lse = torch.logsumexp(logits, -1)
            loss += (lse - logits[rows, labels_part]).sum() / total
            probabilities = torch.exp(logits - lse[:, None])
            probabilities[rows, labels_part] -= 1
            probabilities /= total
            grad_logits = probabilities.to(compute_dtype)
            dx[start:start + chunk] = (grad_logits @ w).to(x.dtype)
            dw.add_((grad_logits.T @ inputs).float())
        ctx.save_for_backward(dx.reshape(shape), dw)
        return loss

    @staticmethod
    def backward(ctx, grad):
        dx, dw = ctx.saved_tensors
        return dx * grad, dw * grad, None, None, None


def training_loss(model, tokens, targets, *, checkpoint_blocks=True, chunk=512):
    """For replicated/DDP model weights; FSDP LM-head has a separate shard path."""
    x = model.token_embeddings(tokens)
    for layer in model.layers:
        x = checkpoint(layer, x, use_reentrant=False) if checkpoint_blocks else layer(x)
    x = model.ln_final(x)
    return ChunkedLinearCrossEntropy.apply(x, model.lm_head.weight, targets, chunk, torch.bfloat16)


class FusedAdamW(torch.optim.Optimizer):
    """Single original Triton pointwise kernel per FP32 master parameter.

    Matches the assignment AdamW epsilon placement: lr*sqrt(1-beta2**t)/
    (1-beta1**t) * m/(sqrt(v)+eps), followed by decoupled weight decay.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        from .triton_optimizer import update
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(m=torch.zeros_like(p), v=torch.zeros_like(p), t=0)
                state["t"] += 1
                b1, b2 = group["betas"]
                alpha = group["lr"] * math.sqrt(1 - b2 ** state["t"]) / (1 - b1 ** state["t"])
                update(p, p.grad, state["m"], state["v"], b1, b2, alpha, group["lr"] * group["weight_decay"], group["eps"])
        return loss
