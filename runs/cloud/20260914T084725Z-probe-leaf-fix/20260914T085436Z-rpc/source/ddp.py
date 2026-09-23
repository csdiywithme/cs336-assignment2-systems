"""Three progressively improved data-parallel containers."""
import torch
import torch.distributed as dist


class DDP(torch.nn.Module):
    def __init__(self, module, mode="overlap"):
        super().__init__()
        if mode not in ("naive", "flat", "overlap"):
            raise ValueError(mode)
        self.module, self.mode = module, mode
        self.world_size = dist.get_world_size()
        self.handles = []
        self.hooks = []
        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor, src=0)
        if mode == "overlap":
            for p in module.parameters():  # parameters() deduplicates tied weights
                if p.requires_grad:
                    self.hooks.append(p.register_post_accumulate_grad_hook(self._ready))

    def _ready(self, p):
        p.grad.div_(self.world_size)
        self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.mode == "overlap":
            for handle in self.handles:
                handle.wait()
            self.handles.clear()
            return
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        if self.mode == "naive":
            for grad in grads:
                dist.all_reduce(grad)
                grad.div_(self.world_size)
        else:
            # Group by dtype/device so flattening never silently changes precision.
            groups = {}
            for grad in grads:
                groups.setdefault((grad.device, grad.dtype), []).append(grad)
            for values in groups.values():
                flat = torch.cat([g.reshape(-1) for g in values])
                dist.all_reduce(flat)
                flat.div_(self.world_size)
                offset = 0
                for grad in values:
                    grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
                    offset += grad.numel()
