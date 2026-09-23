"""NVTX attribution and saved-storage accounting for the unoptimized model."""
import math
from collections import defaultdict

import torch


def annotate_attention():
    import cs336_basics.model as basics
    from cs336_basics.nn_utils import softmax
    def attention(q, k, v, mask=None):
        with torch.cuda.nvtx.range("scaled_dot_product_attention"):
            with torch.cuda.nvtx.range("attention_scores_matmul"):
                scores = (q @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
            if mask is not None:
                with torch.cuda.nvtx.range("attention_mask"):
                    scores = torch.where(mask, scores, float("-inf"))
            with torch.cuda.nvtx.range("attention_softmax"):
                probabilities = softmax(scores, dim=-1)
            with torch.cuda.nvtx.range("attention_values_matmul"):
                return probabilities @ v
    basics.scaled_dot_product_attention = attention


def saved_tensor_accounting(block, x):
    """Deduplicate storage, exclude parameter/buffer aliases, retain all evidence."""
    parameter_storage = {t.untyped_storage().data_ptr() for t in list(block.parameters()) + list(block.buffers())}
    seen, records = set(), []
    stack = []
    hooks = []
    for name, module in block.named_modules():
        hooks.append(module.register_forward_pre_hook(lambda m, args, name=name: stack.append(name or "block")))
        hooks.append(module.register_forward_hook(lambda m, args, result: stack.pop()))
    def pack(t):
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        excluded, duplicate = ptr in parameter_storage, ptr in seen
        row = {"shape": list(t.shape), "dtype": str(t.dtype), "logical_bytes": t.numel() * t.element_size(),
               "storage_bytes": storage.nbytes(), "storage_ptr": ptr, "parameter_or_buffer": excluded,
               "duplicate": duplicate, "module": stack[-1] if stack else "block",
               "producer": type(t.grad_fn).__name__ if t.grad_fn else "leaf"}
        records.append(row)
        if not excluded:
            seen.add(ptr)
        return t
    try:
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = block(x)
            torch.cuda.synchronize()
            after_forward = torch.cuda.memory_allocated()
            y.sum().backward()
            torch.cuda.synchronize()
            after_backward = torch.cuda.memory_allocated()
        contributing = [r for r in records if not r["parameter_or_buffer"] and not r["duplicate"]]
        by_op = defaultdict(int)
        for r in contributing:
            by_op[r["module"] + ":" + r["producer"]] += r["storage_bytes"]
        total = sum(by_op.values())
        return {"records": records, "unique_saved_bytes": total,
                "top_operations": [{"name": key, "bytes": value, "percent": 100 * value / total}
                                   for key, value in sorted(by_op.items(), key=lambda kv: kv[1], reverse=True)],
                "before_forward_bytes": before, "after_forward_bytes": after_forward,
                "after_backward_bytes": after_backward, "peak_bytes": torch.cuda.max_memory_allocated(),
                "parameter_gradient_bytes": sum(p.grad.numel() * p.grad.element_size() for p in block.parameters() if p.grad is not None),
                "input_gradient_bytes": x.grad.numel() * x.grad.element_size()}
    finally:
        for h in hooks:
            h.remove()
