"""Constant-activation-storage, quadratic-recompute checkpointing sketch.

All prefix checkpoints receive the SAME original input (not N different
intermediates). Non-reentrant checkpointing recomputes needed residuals without
retaining a recursive stack of backward gradient outputs. Model weights, parameter
gradients, and Python checkpoint metadata are excluded from the activation bound.
"""
from torch.utils.checkpoint import checkpoint


def minimum_activation_chain(blocks, x):
    def prefix(value, n):
        if n == 0:
            return value
        if n == 1:
            return blocks[0](value)
        previous = checkpoint(lambda original: prefix(original, n - 1), value, use_reentrant=False)
        return blocks[n - 1](previous)
    return prefix(x, len(blocks))
