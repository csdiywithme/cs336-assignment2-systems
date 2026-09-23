"""Explicit experiment matrices. Each case runs in its own fresh process."""


def make_plan(name, gpus):
    if name == "probe":
        return [{"name": "attention-tests", "pytest": ["tests/test_attention.py"], "timeout": 600},
                {"name": "extended-attention", "kind": "validate_attention", "timeout": 600},
                {"name": "precision", "kind": "precision"}]
    if name == "distributed-tests":
        return [{"name": f"distributed-repeat-{i}", "pytest": ["tests/test_ddp.py", "tests/test_fsdp.py",
                 "tests/test_sharded_optimizer.py"], "timeout": 240} for i in range(5)]
    if name == "models":
        return [{"name": f"{size}-{dtype}-{mode}-w{warm}", "kind": "model", "size": size,
                 "dtype": dtype, "mode": mode, "warmup": warm, "steps": 10, "timeout": 500}
                for size in ("small", "medium", "large", "xl", "10B")
                for dtype, warm in [("fp32", 5), ("bf16", 5), ("fp32", 0), ("fp32", 1), ("fp32", 2)]
                for mode in ("forward", "backward", "train")]
    if name == "models-compile":
        return [{"name": f"{size}-compiled-{mode}", "kind": "model", "size": size,
                 "dtype": "fp32", "mode": mode, "compile": True, "warmup": 5, "steps": 10, "timeout": 600}
                for size in ("small", "medium", "large", "xl", "10B") for mode in ("forward", "backward")]
    if name == "attention":
        return [{"name": f"attention-{impl}-s{seq}-d{d}", "kind": "attention", "implementation": impl,
                 "batch": 8, "seq": seq, "dim": d, "dtype": "fp32", "causal": False,
                 "steps": 100, "warmup": 5, "timeout": 300}
                for impl in ("eager", "compiled") for seq in (256, 1024, 4096, 8192, 16384)
                for d in (16, 32, 64, 128)]
    if name == "flash":
        return [{"name": f"flash-{impl}-{dtype}-s{seq}-d{d}", "kind": "attention",
                 "implementation": impl, "batch": 1, "seq": seq, "dim": d, "dtype": dtype,
                 "causal": True, "triton_bench": True, "steps": 10, "warmup": 5, "timeout": 240}
                for impl in ("eager", "triton", "triton-full") for dtype in ("fp32", "bf16")
                for seq in [2**i for i in range(7, 17)] for d in (16, 32, 64, 128)]
    if name == "communication":
        return [{"name": f"{backend}-{gpus}ranks-{mb}MB", "kind": "allreduce", "world": gpus,
                 "backend": backend, "bytes": mb * 1_000_000, "steps": 20, "warmup": 5, "timeout": 240}
                for backend in ("gloo", "nccl") for mb in (1, 10, 100, 1000)]
    if name == "distributed-xl":
        return [{"name": f"xl-{mode}-{opt}", "kind": "distributed", "world": gpus, "size": "xl",
                 "strategy": mode, "optimizer": opt, "warmup": 5, "steps": 10, "timeout": 600}
                for mode, opt in [("naive", "adamw"), ("flat", "adamw"), ("overlap", "adamw"),
                                  ("overlap", "sharded"), ("fsdp", "adamw")]]
    if name == "memory":
        return [{"name": f"memory-xl-s{seq}-{dtype}-{mode}", "kind": "model", "size": "xl",
                 "seq": seq, "dtype": dtype, "mode": mode, "memory_history": True,
                 "warmup": 2, "steps": 2, "timeout": 600}
                for seq in (128, 2048) for dtype in ("fp32", "bf16") for mode in ("forward", "train")]
    if name == "checkpoint":
        return [{"name": f"checkpoint-{segments}", "kind": "model", "size": "xl", "seq": 2048,
                 "dtype": "fp32", "mode": "backward", "checkpoint_segments": segments,
                 "warmup": 2, "steps": 3, "timeout": 600} for segments in (1, 2, 4, 8, 16, 32, 64)]
    if name == "precision":
        return [{"name": "precision", "kind": "precision"}]
    if name == "profile-probe":
        return [{"name": "nsys-small-s256", "kind": "model", "size": "small", "seq": 256,
                 "mode": "train", "dtype": "fp32", "warmup": 5, "steps": 1, "nsys": True, "timeout": 300}]
    if name == "saved-block":
        return [{"name": "saved-block-xl-s2048", "kind": "saved_block", "seq": 2048}]
    if name == "memory-and-checkpoint":
        return make_plan("memory", gpus) + make_plan("checkpoint", gpus) + make_plan("saved-block", gpus)
    if name == "profile-fit":
        return [{"name": f"fit-{size}-{seq}", "kind": "model", "size": size, "seq": seq,
                 "mode": "train", "dtype": "fp32", "warmup": 1, "steps": 1, "timeout": 300}
                for size, seqs in [("small", [4096, 8192, 16384]), ("xl", [1024, 2048, 4096])]
                for seq in seqs]
    if name == "communication-and-xl":
        return make_plan("communication", gpus) + make_plan("distributed-xl", gpus)
    if name == "nsys-probe-and-fit":
        return make_plan("profile-probe", gpus) + make_plan("profile-fit", gpus)
    if name == "optimized-validation":
        return [{"name": "optimized-validation", "kind": "validate_optimized", "timeout": 300}] + make_plan("saved-block", gpus)
    if name == "leaderboard":
        return [{"name": "leaderboard-8b-32768", "kind": "leaderboard", "seq": 32768,
                 "warmup": 2, "steps": 3, "chunk": 512, "timeout": 590}]
    if name == "profiles":
        return [{"name": f"profile-{size}-s{seq}-{'nsys' if nsys else 'plain'}", "kind": "model",
                 "size": size, "seq": seq, "mode": "train", "dtype": "fp32", "annotate": True,
                 "nsys": nsys, "warmup": 5, "steps": 1 if nsys else 5, "timeout": 300}
                for size, seqs in [("small", [256, 1024, 4096]), ("xl", [256, 512, 1024])]
                for seq in seqs for nsys in (False, True)] + [
                    {"name": "profile-saved-block-xl-s2048", "kind": "saved_block", "seq": 2048, "nsys": True}]
    if name == "distributed-profiles":
        return [{"name": f"profile-xl-{strategy}", "kind": "distributed", "world": 2,
                 "size": "xl", "strategy": strategy, "optimizer": "adamw", "nsys": True,
                 "warmup": 5, "steps": 1, "timeout": 300} for strategy in ("naive", "overlap", "fsdp")]
    raise ValueError(f"Unknown plan {name}")
