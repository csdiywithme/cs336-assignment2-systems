"""Additional regressions; original course tests remain unchanged."""
from copy import deepcopy
import tempfile

import pytest
import torch
import torch.distributed as dist

from cs336_systems.checkpointing import minimum_activation_chain
from cs336_systems.sharded_optimizer import ShardedOptimizer


def test_prefix_checkpoint_gradients():
    torch.manual_seed(3)
    modules = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.SiLU()) for _ in range(8)])
    x = torch.randn(4, 16, requires_grad=True)
    y = x
    for block in modules:
        y = block(y)
    reference = torch.autograd.grad(y.sum(), (x, *modules.parameters()))
    actual = torch.autograd.grad(minimum_activation_chain(modules, x).sum(), (x, *modules.parameters()))
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_optimizer_groups_add_and_resume():
    with tempfile.TemporaryDirectory(prefix="cs336-a2-group-test-") as temp:
        dist.init_process_group("gloo", init_method=f"file://{temp}/rendezvous", rank=0, world_size=1)
        try:
            torch.manual_seed(9)
            p = torch.nn.Parameter(torch.randn(13))
            q = torch.nn.Parameter(torch.randn(7))
            p2, q2 = torch.nn.Parameter(p.detach().clone()), torch.nn.Parameter(q.detach().clone())
            opt = ShardedOptimizer([{"params": [p], "lr": 0.01}], torch.optim.AdamW, weight_decay=0.1)
            ref = torch.optim.AdamW([{"params": [p2], "lr": 0.01}], weight_decay=0.1)
            opt.add_param_group({"params": [q], "lr": 0.003, "weight_decay": 0.0})
            ref.add_param_group({"params": [q2], "lr": 0.003, "weight_decay": 0.0})
            for i in range(6):
                for a, b in ((p, p2), (q, q2)):
                    a.grad = torch.randn_like(a)
                    b.grad = a.grad.clone()
                opt.step()
                ref.step()
                if i == 2:
                    state = deepcopy(opt.state_dict())
                    opt = ShardedOptimizer([{"params": [p], "lr": 0.01},
                                            {"params": [q], "lr": 0.003, "weight_decay": 0.0}],
                                           torch.optim.AdamW, weight_decay=0.1)
                    opt.load_state_dict(state)
                torch.testing.assert_close(p, p2, atol=0, rtol=0)
                torch.testing.assert_close(q, q2, atol=0, rtol=0)
            opt.zero_grad(set_to_none=True)
            assert p.grad is None and q.grad is None
        finally:
            dist.destroy_process_group()
