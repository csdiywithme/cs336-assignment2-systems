"""Fresh-process benchmark worker; every measurement is written as JSON.

Run: python -m cs336_systems.experiments --config '{...}' --output runs/example
Timing uses synchronized perf_counter, except the explicitly labelled Triton
microbenchmarks. Out-of-memory is an observation, never a substituted estimate.
"""
import argparse
import contextlib
import datetime
import gc
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import torch

SIZES = {"small": (768, 3072, 12, 12), "medium": (1024, 4096, 24, 16),
         "large": (1280, 5120, 36, 20), "xl": (2560, 10240, 32, 32),
         "10B": (4608, 12288, 50, 36), "leaderboard": (4096, 11008, 34, 32)}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory():
    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def stats(samples):
    return {"samples_ms": samples, "mean_ms": statistics.mean(samples),
            "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "median_ms": statistics.median(samples)}


def timed(fn, warmup, steps):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(steps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return stats(times)


def model_factory(cfg):
    from cs336_basics.model import BasicsTransformerLM
    d, ff, layers, heads = SIZES[cfg.get("size", "small")]
    with torch.device("cuda"):
        model = BasicsTransformerLM(vocab_size=cfg.get("vocab", 10000), context_length=cfg.get("seq", 512),
                                    d_model=d, num_layers=layers, num_heads=heads, d_ff=ff)
    if cfg.get("checkpoint_segments"):
        from types import MethodType
        from torch.utils.checkpoint import checkpoint
        segments = cfg["checkpoint_segments"]
        def checkpoint_forward(self, tokens):
            x = self.token_embeddings(tokens)
            chunk = math.ceil(len(self.layers) / segments)
            for start in range(0, len(self.layers), chunk):
                group = tuple(self.layers[start:start + chunk])
                def run(value, layers=group):
                    for layer in layers:
                        value = layer(value)
                    return value
                # Checkpoint every segment including the last: one full recompute.
                x = checkpoint(run, x, use_reentrant=False)
            return self.lm_head(self.ln_final(x))
        model.forward = MethodType(checkpoint_forward, model)
    return model


def model_benchmark(cfg, out):
    from cs336_basics.optimizer import AdamW
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model_factory(cfg)
    parameter_count = sum(p.numel() for p in model.parameters())
    opt = AdamW(model.parameters(), lr=1e-3) if cfg["mode"] == "train" else None
    if cfg.get("compile"):
        model = torch.compile(model)
    batch, seq, vocab = cfg.get("batch", 4), cfg.get("seq", 512), cfg.get("vocab", 10000)
    x = torch.randint(vocab, (batch, seq), device="cuda")
    y = torch.randint(vocab, (batch, seq), device="cuda")
    dtype = torch.bfloat16 if cfg.get("dtype") == "bf16" else torch.float32
    stages = []
    def step(record=False):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            with torch.cuda.nvtx.range("forward"):
                logits = model(x)
            if cfg["mode"] != "forward":
                with torch.cuda.nvtx.range("loss"):
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        if record:
            stages.append({"stage": "after_forward", **memory()})
        if cfg["mode"] != "forward":
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
            if record:
                stages.append({"stage": "after_backward_before_optimizer", **memory()})
        if opt is not None:
            with torch.cuda.nvtx.range("optimizer"):
                opt.step()
            if record:
                stages.append({"stage": "after_optimizer", **memory()})
        return logits[0, 0, 0].detach() if cfg["mode"] == "forward" else loss.detach()
    initial = memory()
    try:
        for _ in range(cfg.get("warmup", 5)):
            step()
        sync()
        if cfg.get("memory_history"):
            torch.cuda.memory._record_memory_history(max_entries=500000)
        with torch.cuda.nvtx.range("measurement"):
            result = timed(step, 0, cfg.get("steps", 10))
        model.zero_grad(set_to_none=True)
        sync()
        torch.cuda.reset_peak_memory_stats()
        loss = step(record=True)
        sync()
        result.update({"last_scalar": loss.item(), "parameters": parameter_count, "initial_memory": initial,
                       "memory": memory(), "stages": stages})
        if cfg.get("memory_history"):
            torch.cuda.memory._dump_snapshot(str(out / "memory_snapshot.pickle"))
        return result
    finally:
        if cfg.get("memory_history"):
            torch.cuda.memory._record_memory_history(enabled=None)


def validate_attention(cfg, out):
    from .attention import attention_reference, FlashAttentionTriton, FlashAttentionTritonFull
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for n, d in ((32, 16), (128, 64), (256, 128), (63, 32)):
            for causal in (False, True):
                torch.manual_seed(0)
                base = [torch.randn(2, n, d, device="cuda", dtype=dtype) for _ in range(3)]
                do = torch.randn_like(base[0])
                refs = [v.detach().float().clone().requires_grad_() for v in base]
                expected = attention_reference(*refs, causal)
                expected.backward(do.float())
                for cls in (FlashAttentionTriton, FlashAttentionTritonFull):
                    inputs = [v.detach().clone().requires_grad_() for v in base]
                    output = cls.apply(*inputs, causal)
                    output.backward(do)
                    errors = [float((output.float() - expected).abs().max())]
                    errors += [float((a.grad.float() - b.grad).abs().max()) for a, b in zip(inputs, refs)]
                    atol = 0.03 if dtype == torch.bfloat16 else 1e-3
                    for actual, target in [(output, expected), *[(a.grad, b.grad) for a, b in zip(inputs, refs)]]:
                        torch.testing.assert_close(actual.float(), target, atol=atol, rtol=0.03)
                    records.append({"implementation": cls.__name__, "dtype": str(dtype), "n": n, "d": d,
                                    "causal": causal, "max_abs_errors_output_dq_dk_dv": errors})
                    print(json.dumps(records[-1]), flush=True)
    return {"checks": records}


def attention_benchmark(cfg, out):
    from .attention import attention_reference, FlashAttentionTriton, FlashAttentionTritonFull
    torch.manual_seed(2026)
    dtype = torch.bfloat16 if cfg["dtype"] == "bf16" else torch.float32
    inputs = [torch.randn(cfg["batch"], cfg["seq"], cfg["dim"], device="cuda", dtype=dtype,
                          requires_grad=True) for _ in range(3)]
    do = torch.randn_like(inputs[0])
    impl = cfg["implementation"]
    function = {"eager": attention_reference, "compiled": torch.compile(attention_reference),
                "triton": FlashAttentionTriton.apply, "triton-full": FlashAttentionTritonFull.apply}[impl]
    def forward():
        return function(*inputs, cfg["causal"])
    def end_to_end():
        for t in inputs:
            t.grad = None
        forward().backward(do)
    result = {}
    def persist():
        (out / "partial.json").write_text(json.dumps(result, indent=2))
    if cfg.get("triton_bench"):
        from triton.testing import do_bench
        bench = lambda fn: {"mean_ms": do_bench(fn, warmup=100, rep=300, return_mode="mean"),
                            "method": "triton.testing.do_bench", "warmup_ms": 100, "rep_ms": 300}
    else:
        bench = lambda fn: timed(fn, cfg["warmup"], cfg["steps"])
    result["forward"] = bench(forward)
    persist()
    output = forward()
    sync()
    result["memory_before_backward"] = memory()
    # Retain exactly one fixed forward graph: backward-only timings exclude forward.
    def backward():
        for t in inputs:
            t.grad = None
        output.backward(do, retain_graph=True)
    result["backward"] = bench(backward)
    persist()
    del output
    gc.collect()
    torch.cuda.empty_cache()
    result["end_to_end"] = bench(end_to_end)
    result["memory"] = memory()
    return result


def precision_experiment(cfg, out):
    values = []
    for accum, source in ((torch.float32, torch.float32), (torch.float16, torch.float16),
                           (torch.float32, torch.float16)):
        total = torch.tensor(0., dtype=accum, device="cuda")
        increment = torch.tensor(0.01, dtype=source, device="cuda")
        for _ in range(1000):
            total += increment
        values.append({"accumulator": str(accum), "increment": str(source), "value": total.item()})
    total = torch.tensor(0., dtype=torch.float32, device="cuda")
    for _ in range(1000):
        total += torch.tensor(0.01, dtype=torch.float16, device="cuda").float()
    values.append({"variant": "explicit_cast", "value": total.item()})
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10, bias=False)
            self.ln = torch.nn.LayerNorm(10)
            self.fc2 = torch.nn.Linear(10, 10, bias=False)
        def forward(self, x):
            a = self.fc1(x)
            b = self.ln(torch.relu(a))
            c = self.fc2(b)
            return a, b, c
    net = Toy().cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        a, b, c = net(torch.randn(4, 10, device="cuda"))
        loss = torch.nn.functional.cross_entropy(c, torch.arange(4, device="cuda"))
    loss.backward()
    return {"accumulations": values, "autocast": {"fc1": str(a.dtype), "layernorm": str(b.dtype),
            "logits": str(c.dtype), "loss": str(loss.dtype),
            "parameters": {n: str(p.dtype) for n, p in net.named_parameters()},
            "gradients": {n: str(p.grad.dtype) for n, p in net.named_parameters()}}}


def distributed_worker(rank, world, cfg, output_dir, port):
    import os
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    backend = cfg.get("backend", "nccl")
    dist.init_process_group(backend, rank=rank, world_size=world)
    out = Path(output_dir)
    try:
        if cfg["kind"] == "allreduce":
            device = "cpu" if backend == "gloo" else f"cuda:{rank}"
            tensor = torch.ones(cfg["bytes"] // 4, device=device)
            def step():
                # Reset avoids overflow across repeated sum reductions.
                tensor.fill_(rank + 1)
                sync()
                dist.barrier()
                start = time.perf_counter()
                dist.all_reduce(tensor)
                sync()
                return (time.perf_counter() - start) * 1000
            for _ in range(cfg["warmup"]):
                step()
            samples = [step() for _ in range(cfg["steps"])]
            torch.testing.assert_close(tensor[0], torch.tensor(world * (world + 1) / 2, device=device))
            result = stats(samples)
        else:
            from cs336_basics.optimizer import AdamW
            from .ddp import DDP
            from .fsdp import FSDP
            from .sharded_optimizer import ShardedOptimizer
            torch.manual_seed(2026)
            model = model_factory(cfg)
            strategy = cfg["strategy"]
            wrapper = FSDP(model, compute_dtype=torch.bfloat16 if cfg.get("dtype") == "bf16" else None) if strategy == "fsdp" else DDP(model, mode=strategy)
            opt = (ShardedOptimizer(wrapper.parameters(), AdamW, lr=1e-3)
                   if cfg.get("optimizer") == "sharded" else AdamW(wrapper.parameters(), lr=1e-3))
            x = torch.randint(10000, (cfg.get("batch", 4) // world, cfg.get("seq", 512)), device="cuda")
            y = torch.randint(10000, x.shape, device="cuda")
            result = {"memory_after_initialization": memory()}
            samples, communication_samples, stages = [], [], []
            for i in range(cfg["warmup"] + cfg["steps"]):
                wrapper.zero_grad(set_to_none=True)
                dist.barrier()
                sync()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.cuda.nvtx.range("forward"):
                    logits = wrapper(x)
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
                with torch.cuda.nvtx.range("backward"):
                    loss.backward()
                sync()
                communication_start = time.perf_counter()
                with torch.cuda.nvtx.range("gradient_sync_wait"):
                    wrapper.finish_gradient_synchronization()
                    sync()
                exposed = (time.perf_counter() - communication_start) * 1000
                before_opt = memory()
                with torch.cuda.nvtx.range("optimizer"):
                    opt.step()
                sync()
                duration = (time.perf_counter() - start) * 1000
                if i >= cfg["warmup"]:
                    samples.append(duration)
                    communication_samples.append(exposed)
                    stages.append({"before_optimizer": before_opt, "after_optimizer": memory()})
            result.update({"step": stats(samples), "exposed_sync": stats(communication_samples),
                           "memory_stages": stages, "loss": loss.item(),
                           "communication_note": "Exposed wait only; overlapped traffic is not counted as zero traffic."})
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            (out / "ranks.json").write_text(json.dumps(gathered, indent=2))
    finally:
        dist.destroy_process_group()


def distributed(cfg, out):
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(distributed_worker, args=(cfg["world"], cfg, str(out), port), nprocs=cfg["world"], join=True)
    ranks = json.loads((out / "ranks.json").read_text())
    if cfg["kind"] == "allreduce":
        # A collective iteration completes at its slowest rank.
        max_times = [max(r["samples_ms"][i] for r in ranks) for i in range(cfg["steps"])]
        aggregate = stats(max_times)
        aggregate["bus_bandwidth_GBps"] = 2 * (cfg["world"] - 1) / cfg["world"] * cfg["bytes"] / (aggregate["mean_ms"] * 1e6)
        return {"ranks": ranks, "rank_max": aggregate}
    return {"ranks": ranks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, out = json.loads(args.config), args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    result = {"config": cfg, "started_utc": datetime.datetime.now(datetime.UTC).isoformat()}
    kinds = {"model": model_benchmark, "attention": attention_benchmark, "validate_attention": validate_attention,
             "precision": precision_experiment, "allreduce": distributed, "distributed": distributed}
    try:
        result["measurements"] = kinds[cfg["kind"]](cfg, out)
        result["status"] = "ok"
    except torch.OutOfMemoryError:
        result.update(status="oom", traceback=traceback.format_exc(), memory=memory())
    except Exception:
        result.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"] = time.time() - start
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
