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
    d, ff = cfg.get("d_model", d), cfg.get("d_ff", ff)
    layers, heads = cfg.get("num_layers", layers), cfg.get("num_heads", heads)
    with torch.device("cuda"):
        model = BasicsTransformerLM(vocab_size=cfg.get("vocab", 10000), context_length=cfg.get("seq", 512),
                                    d_model=d, num_layers=layers, num_heads=heads, d_ff=ff)
    if cfg.get("checkpoint_segments"):
        from types import MethodType
        from torch.utils.checkpoint import checkpoint
        segments = cfg["checkpoint_segments"]
        def checkpoint_forward(self, tokens):
            x = self.token_embeddings(tokens)
            if segments == 2 * len(self.layers):
                for layer in self.layers:
                    def attention_half(value, layer=layer):
                        return value + layer.attn(layer.ln1(value))
                    def ffn_half(value, layer=layer):
                        return value + layer.ffn(layer.ln2(value))
                    x = checkpoint(attention_half, x, use_reentrant=False)
                    x = checkpoint(ffn_half, x, use_reentrant=False)
                return self.lm_head(self.ln_final(x))
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
    if cfg.get("nsys") or cfg.get("annotate"):
        from .profiling import annotate_attention
        annotate_attention()
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
    stage_events = []
    def step(record=False):
        model.zero_grad(set_to_none=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        events[0].record()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            with torch.cuda.nvtx.range("forward"):
                logits = model(x)
            events[1].record()
            if cfg["mode"] != "forward":
                with torch.cuda.nvtx.range("loss"):
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
            events[2].record()
        if record:
            stages.append({"stage": "after_forward", **memory()})
        if cfg["mode"] != "forward":
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
            events[3].record()
            if record:
                stages.append({"stage": "after_backward_before_optimizer", **memory()})
        if opt is not None:
            with torch.cuda.nvtx.range("optimizer"):
                opt.step()
            events[4].record()
            if record:
                stages.append({"stage": "after_optimizer", **memory()})
        stage_events.append(events)
        return logits[0, 0, 0].detach() if cfg["mode"] == "forward" else loss.detach()
    initial = memory()
    try:
        for _ in range(cfg.get("warmup", 5)):
            step()
        sync()
        stage_events.clear()
        if cfg.get("memory_history"):
            torch.cuda.memory._record_memory_history(max_entries=500000)
        with torch.cuda.nvtx.range("measurement"):
            result = timed(step, 0, cfg.get("steps", 10))
        event_results = {"forward": stats([e[0].elapsed_time(e[1]) for e in stage_events])}
        if cfg["mode"] != "forward":
            event_results["loss"] = stats([e[1].elapsed_time(e[2]) for e in stage_events])
            event_results["backward"] = stats([e[2].elapsed_time(e[3]) for e in stage_events])
        if opt is not None:
            event_results["optimizer"] = stats([e[3].elapsed_time(e[4]) for e in stage_events])
        result["cuda_event_stages"] = event_results
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
                if cfg.get("nsys") and i == cfg["warmup"]:
                    torch.cuda.nvtx.range_push("measurement")
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
                if cfg.get("nsys") and i == cfg["warmup"]:
                    torch.cuda.nvtx.range_pop()
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


def saved_block(cfg, out):
    from cs336_basics.model import TransformerBlock, RotaryEmbedding
    from .profiling import saved_tensor_accounting
    with torch.device("cuda"):
        block = TransformerBlock(2560, 32, 10240, RotaryEmbedding(cfg.get("seq", 2048), 80))
    x = torch.randn(4, cfg.get("seq", 2048), 2560, device="cuda", requires_grad=True)
    with torch.cuda.nvtx.range("measurement"):
        return saved_tensor_accounting(block, x)


def validate_optimized(cfg, out):
    from copy import deepcopy
    from cs336_basics.optimizer import AdamW
    from .optimized_model import install_flash_attention, ChunkedLinearCrossEntropy, FusedAdamW
    torch.manual_seed(7)
    tiny = {"size": "small", "d_model": 128, "d_ff": 256, "num_layers": 2, "num_heads": 4, "vocab": 256, "seq": 64}
    reference = model_factory(tiny)
    optimized = install_flash_attention(deepcopy(reference), tuned=cfg.get("tuned", False))
    tokens = torch.randint(256, (2, 64), device="cuda")
    targets = torch.randint(256, (2, 64), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = reference(tokens), optimized(tokens)
        la = torch.nn.functional.cross_entropy(a.flatten(0, 1).float(), targets.flatten())
        lb = torch.nn.functional.cross_entropy(b.flatten(0, 1).float(), targets.flatten())
    la.backward()
    lb.backward()
    torch.testing.assert_close(a.float(), b.float(), atol=0.025, rtol=0.03)
    grad_errors = {}
    for (name, p), (_, q) in zip(reference.named_parameters(), optimized.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0.001, rtol=0.1)
        grad_errors[name] = (p.grad-q.grad).abs().max().item()
    x = torch.randn(2, 19, 128, device="cuda", requires_grad=True)
    w = torch.randn(257, 128, device="cuda", requires_grad=True)
    y = torch.randint(257, (2, 19), device="cuda")
    xc, wc = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ordinary = torch.nn.functional.cross_entropy((x @ w.T).flatten(0, 1).float(), y.flatten())
        chunked = ChunkedLinearCrossEntropy.apply(xc, wc, y, 8, torch.bfloat16)
    ordinary.backward()
    chunked.backward()
    # Different GEMM M dimensions can select different BF16 reductions. Check
    # BF16 at the handout tolerance, and separately test the algebra in FP32.
    torch.testing.assert_close(ordinary, chunked, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(x.grad, xc.grad, atol=0.006, rtol=0.03)
    torch.testing.assert_close(w.grad, wc.grad, atol=0.006, rtol=0.03)
    xf, wf = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    xf2, wf2 = xf.detach().clone().requires_grad_(), wf.detach().clone().requires_grad_()
    reference_fp32 = torch.nn.functional.cross_entropy((xf @ wf.T).flatten(0, 1), y.flatten())
    chunked_fp32 = ChunkedLinearCrossEntropy.apply(xf2, wf2, y, 8, torch.float32)
    reference_fp32.backward()
    chunked_fp32.backward()
    torch.testing.assert_close(reference_fp32, chunked_fp32, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xf.grad, xf2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(wf.grad, wf2.grad, atol=1e-5, rtol=1e-5)
    p = torch.nn.Parameter(torch.randn(12345, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    o1, o2 = AdamW([p]), FusedAdamW([q])
    for _ in range(10):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        o1.step()
        o2.step()
    torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)
    return {"model_max_output_error": (a-b).abs().max().item(), "model_gradient_errors": grad_errors,
            "loss_reference": ordinary.item(), "loss_chunked": chunked.item(),
            "chunked_dx_max_error": (x.grad-xc.grad).abs().max().item(),
            "chunked_dw_max_error": (w.grad-wc.grad).abs().max().item(),
            "fp32_loss_error": (reference_fp32-chunked_fp32).abs().item(),
            "fp32_dx_error": (xf.grad-xf2.grad).abs().max().item(),
            "fp32_dw_error": (wf.grad-wf2.grad).abs().max().item(),
            "adamw_10_step_max_error": (p-q).abs().max().item()}


def leaderboard_worker(rank, world, cfg, output_dir, port):
    import os
    import torch.distributed as dist
    from .ddp import DDP
    from .sharded_optimizer import ShardedOptimizer
    from .optimized_model import install_flash_attention, training_loss, FusedAdamW
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        started = time.perf_counter()
        torch.manual_seed(2026)
        model = install_flash_attention(model_factory({"size": "leaderboard", "seq": cfg.get("seq", 32768), "vocab": 151936}),
                                        tuned=cfg.get("tuned", False))
        wrapper = DDP(model, mode="overlap")
        optimizer = ShardedOptimizer(wrapper.parameters(), FusedAdamW)
        torch.manual_seed(2026 + rank)
        tokens = torch.randint(151936, (1, cfg.get("seq", 32768)), device="cuda")
        targets = torch.randint(151936, tokens.shape, device="cuda")
        measurements = []
        for i in range(cfg.get("warmup", 2) + cfg.get("steps", 3)):
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            sync()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = training_loss(model, tokens, targets, checkpoint_blocks=True, chunk=cfg.get("chunk", 512))
            loss.backward()
            wrapper.finish_gradient_synchronization()
            optimizer.step()
            sync()
            row = {"iteration": i, "warmup": i < cfg.get("warmup", 2),
                   "step_ms": (time.perf_counter()-start)*1000, "loss": loss.item(), **memory()}
            measurements.append(row)
            print(f"rank={rank} " + json.dumps(row), flush=True)
            (Path(output_dir) / f"rank{rank}-progress.json").write_text(json.dumps(measurements, indent=2))
        result = {"measurements": measurements, "cold_start_total_seconds": time.perf_counter()-started,
                  "parameters": sum(p.numel() for p in model.parameters())}
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            (Path(output_dir) / "leaderboard-ranks.json").write_text(json.dumps(gathered, indent=2))
    finally:
        dist.destroy_process_group()


def leaderboard(cfg, out):
    import os
    import socket
    import torch.multiprocessing as mp
    # Fresh, unique, initially nonexistent cache directories for the timed run.
    os.environ["TRITON_CACHE_DIR"] = str(out / "cold-triton-cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "cold-inductor-cache")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(leaderboard_worker, args=(2, cfg, str(out), port), nprocs=2, join=True)
    ranks = json.loads((out / "leaderboard-ranks.json").read_text())
    times = [max(r["measurements"][i]["step_ms"] for r in ranks)
             for i in range(cfg.get("warmup", 2), cfg.get("warmup", 2)+cfg.get("steps", 3))]
    return {"rank_max_step": stats(times), "ranks": ranks,
            "loss_reduction": "global token mean; equal local batch/token counts on both ranks",
            "timing": "Synchronized wall-clock full forward, CE, backward, gradient all-reduce, AdamW, owner broadcast"}


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
             "precision": precision_experiment, "allreduce": distributed, "distributed": distributed,
             "saved_block": saved_block}
    kinds.update(validate_optimized=validate_optimized, leaderboard=leaderboard)
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
