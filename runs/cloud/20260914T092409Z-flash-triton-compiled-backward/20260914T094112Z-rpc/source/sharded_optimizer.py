"""Whole-parameter optimizer-state sharding, followed by owner broadcasts."""
import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.optimizer_cls, self.optimizer_kwargs = optimizer_cls, kwargs
        self.owners = {}
        self.loads = [0] * self.world_size
        self.local_optimizer = None
        super().__init__(params, defaults=dict(kwargs))
        # A group with no local params is valid and permits empty-rank shards.
        groups = [self._local_group(g) for g in self.param_groups]
        self.local_optimizer = optimizer_cls(groups, **kwargs)
        self.state = self.local_optimizer.state
        self.defaults = dict(self.local_optimizer.defaults)
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)

    def _local_group(self, group):
        return {**group, "params": [p for p in group["params"] if self.owners[p] == self.rank]}

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for p in group["params"]:
            if p not in self.owners:
                owner = min(range(self.world_size), key=lambda r: self.loads[r])
                self.owners[p] = owner
                self.loads[owner] += p.numel()
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(self._local_group(group))

    def state_dict(self):
        result = super().state_dict()
        result["shard_metadata"] = {"rank": self.rank, "world_size": self.world_size}
        return result

    def load_state_dict(self, state_dict):
        metadata = state_dict.get("shard_metadata", {})
        if metadata and metadata != {"rank": self.rank, "world_size": self.world_size}:
            raise ValueError("Load the checkpoint for this rank and world size; resharding is not implicit")
        super().load_state_dict(state_dict)
        self.local_optimizer.state = self.state
        self.local_optimizer.param_groups = [self._local_group(g) for g in self.param_groups]

    def step(self, closure=None, **kwargs):
        # Respect scheduler changes to the public parameter groups.
        for public, local in zip(self.param_groups, self.local_optimizer.param_groups):
            local.update({k: v for k, v in public.items() if k != "params"})
        loss = self.local_optimizer.step(closure=closure, **kwargs)
        with torch.no_grad():
            handles = [dist.broadcast(p, src=owner, async_op=True) for p, owner in self.owners.items()]
            for handle in handles:
                handle.wait()
        return loss
