"""Layer-wise FSDP with FP32 flat master shards and bounded weight prefetch.

Custom autograd avoids modifying Parameter storage while autograd is using it.
Original module types/names remain intact, including the assignment's own Linear.
"""
import math
from types import MethodType

import torch
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear, SwiGLU


class _Shard:
    def __init__(self, parameter, owner):
        self.parameter, self.owner = parameter, owner
        self.shape = tuple(parameter.shape)
        self.numel = parameter.numel()
        self.size = math.ceil(self.numel / owner.world_size)
        flat = torch.zeros(self.size * owner.world_size, device=parameter.device, dtype=torch.float32)
        flat[:self.numel].copy_(parameter.detach().flatten())
        parameter.data = flat.narrow(0, owner.rank * self.size, self.size).clone()
        self.full = self.handle = self.send = None

    def prefetch(self):
        if self.full is None:
            self.send = self.parameter.detach().to(self.owner.compute_dtype or torch.float32).contiguous()
            self.full = torch.empty(self.size * self.owner.world_size, device=self.send.device, dtype=self.send.dtype)
            self.handle = dist.all_gather_into_tensor(self.full, self.send, async_op=True)

    def gather(self):
        self.prefetch()
        self.handle.wait()
        return self.full[:self.numel].view(self.shape)

    def release(self):
        self.full = self.handle = self.send = None

    def reduce_gradient(self, gradient):
        full = torch.zeros(self.size * self.owner.world_size, device=gradient.device, dtype=torch.float32)
        full[:self.numel].copy_(gradient.flatten())
        full.div_(self.owner.world_size)
        result = torch.empty_like(self.parameter)
        handle = dist.reduce_scatter_tensor(result, full, async_op=True)
        return result, handle, full


class _ShardedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, shard):
        full = shard.gather()
        # Mirrors autocast for mixed-precision Linear/Embedding weights.
        dtype = shard.owner.compute_dtype or (torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype)
        xc, wc = x.to(dtype), full.to(dtype)
        output = xc @ wc.T
        if bias is not None:
            output = output + bias.to(dtype)
        ctx.save_for_backward(xc, weight)
        ctx.shard, ctx.input_dtype, ctx.has_bias = shard, x.dtype, bias is not None
        shard.release()
        shard.owner.after_forward(shard)
        return output

    @staticmethod
    def backward(ctx, do):
        x, weight = ctx.saved_tensors
        shard = ctx.shard
        full = shard.gather().to(x.dtype)
        go = do.to(x.dtype)
        dw = go.reshape(-1, go.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        grad, handle, communication_input = shard.reduce_gradient(dw)
        dx = (go @ full).to(ctx.input_dtype)
        db = go.reshape(-1, go.shape[-1]).sum(0).float() if ctx.has_bias else None
        handle.wait()
        shard.release()
        return dx, grad, db, None


class _ShardedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight, shard, padding_idx):
        full = shard.gather()
        out = torch.nn.functional.embedding(ids, full, padding_idx=padding_idx)
        ctx.save_for_backward(ids, weight)
        ctx.shard, ctx.padding_idx = shard, padding_idx
        shard.release()
        shard.owner.after_forward(shard)
        return out

    @staticmethod
    def backward(ctx, do):
        ids, weight = ctx.saved_tensors
        shard = ctx.shard
        # Embedding backward needs indices and dO, not the embedding weights.
        dw = torch.ops.aten.embedding_dense_backward(do, ids, shard.shape[0], ctx.padding_idx, False)
        grad, handle, communication_input = shard.reduce_gradient(dw)
        handle.wait()
        return None, grad, None, None


class FSDP(torch.nn.Module):
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.shards, self.by_parameter, self.handles, self.hooks = [], {}, [], []
        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor, src=0)
        ordered_modules = list(module.modules())
        # Staff SwiGLU registers w1,w2,w3 but executes w1,w3,w2.
        # Prefetch must follow execution, not registration order.
        for parent in module.modules():
            if isinstance(parent, SwiGLU):
                i, j = ordered_modules.index(parent.w2), ordered_modules.index(parent.w3)
                ordered_modules[i], ordered_modules[j] = ordered_modules[j], ordered_modules[i]
        for child in ordered_modules:
            if not isinstance(child, (Linear, Embedding, torch.nn.Linear, torch.nn.Embedding)):
                continue
            if child.weight not in self.by_parameter:
                shard = _Shard(child.weight, self)
                self.by_parameter[child.weight] = shard
                self.shards.append(shard)
            else:
                shard = self.by_parameter[child.weight]
            child._fsdp_shard = shard
            if isinstance(child, (Linear, torch.nn.Linear)):
                def linear_forward(m, x):
                    return _ShardedLinear.apply(x, m.weight, getattr(m, "bias", None), m._fsdp_shard)
                child.forward = MethodType(linear_forward, child)
            else:
                if getattr(child, "max_norm", None) is not None or getattr(child, "scale_grad_by_freq", False) or getattr(child, "sparse", False):
                    raise ValueError("Assignment FSDP supports dense, unnormalized Embedding only")
                def embedding_forward(m, ids):
                    padding = getattr(m, "padding_idx", None)
                    return _ShardedEmbedding.apply(ids, m.weight, m._fsdp_shard, -1 if padding is None else padding)
                child.forward = MethodType(embedding_forward, child)
        self.positions = {s: i for i, s in enumerate(self.shards)}
        for p in module.parameters():
            if p.requires_grad and p not in self.by_parameter:
                self.hooks.append(p.register_post_accumulate_grad_hook(self._replicated_ready))

    def _replicated_ready(self, p):
        p.grad.div_(self.world_size)
        self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def after_forward(self, shard):
        # After i completes, start i+2. At most current/next weight is prefetched;
        # caller execution order is the normal sequential Transformer order.
        index = self.positions[shard] + 2
        if index < len(self.shards):
            self.shards[index].prefetch()

    def forward(self, *args, **kwargs):
        for shard in self.shards[:2]:
            shard.prefetch()
        result = self.module(*args, **kwargs)
        # Release unused prefetched layers (e.g. a conditional module branch).
        for shard in self.shards:
            if shard.full is not None:
                shard.handle.wait()
                shard.release()
        return result

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    @torch.no_grad()
    def gather_full_params(self):
        result = {}
        for name, p in self.module.named_parameters():
            if p in self.by_parameter:
                shard = self.by_parameter[p]
                full = torch.empty(shard.size * self.world_size, device=p.device, dtype=p.dtype)
                dist.all_gather_into_tensor(full, p.contiguous())
                result[name] = full[:shard.numel].view(shard.shape)
            else:
                result[name] = p.detach().clone()
        return result
