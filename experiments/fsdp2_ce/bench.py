"""Two-rank FSDP2 ablation worker. Use torchrun from the A2 directory."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.functional import cross_entropy

from cs336_systems.experiments import model_factory


WORKLOADS = {
    "smoke": {"size": "small", "d_model": 256, "d_ff": 1024, "num_layers": 4,
              "num_heads": 4, "seq": 128, "vocab": 1024},
    "xl": {"size": "xl", "seq": 512, "vocab": 10_000},
}
TOLERANCES = {
    "loss": {"rtol": 2e-6, "atol": 2e-6},
    "gradient": {"rtol": 2e-5, "atol": 1e-7},
    "parameter": {"rtol": 2e-5, "atol": 2e-6},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("default", "symm", "zero"))
    parser.add_argument("--workload", required=True, choices=WORKLOADS)
    parser.add_argument("--mode", required=True, choices=("check", "bench", "trace"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="Default FSDP2 tensor reference directory for check mode")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--forward-prefetch", choices=("default", "next"), default="default")
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1:
        parser.error("--warmup and --steps must be positive")
    if args.mode == "check" and args.reference is None:
        parser.error("--reference is required in check mode")
    return args


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def make_workload(args, device, rank):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    model = model_factory(WORKLOADS[args.workload])
    config = dict(model.config)
    config.update({"global_batch": 4, "dtype": "float32", "activation_checkpointing": False,
                   "tf32": False, "optimizer": "torch.optim.AdamW", "adamw_lr": 1e-3,
                   "adamw_foreach": False, "adamw_fused": False,
                   "reshard_after_forward": {"block": True, "root": False},
                   "parameters": sum(param.numel() for param in model.parameters())})
    generator = torch.Generator().manual_seed(args.seed + 1)
    shape = (config["global_batch"], config["context_length"])
    batch = torch.randint(config["vocab_size"], shape, generator=generator)
    labels = torch.randint(config["vocab_size"], shape, generator=generator)
    config["input_sha256"] = hashlib.sha256(batch.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    inputs = batch.chunk(dist.get_world_size())[rank].to(device)
    targets = labels.chunk(dist.get_world_size())[rank].to(device)

    mesh = DeviceMesh.from_group(dist.group.WORLD, "cuda", mesh_dim_names=("shard",))
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    if args.variant != "default":
        for module in (*model.layers, model):
            module.set_symm_mem_for_comm("NCCL")
    if args.forward_prefetch == "next":
        for layer, successor in zip(model.layers, list(model.layers)[1:]):
            layer.set_modules_to_forward_prefetch([successor])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    return model, optimizer, inputs, targets, config


def forward_loss(model, inputs, targets):
    logits = model(inputs)
    return cross_entropy(logits.flatten(0, 1), targets.flatten())


def measure_training_step(model, optimizer, inputs, targets, annotate=False):
    nvtx = torch.cuda.nvtx.range if annotate else lambda name: nullcontext()
    with nvtx("cebench/zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with nvtx("cebench/forward"):
        loss = forward_loss(model, inputs, targets)
    with nvtx("cebench/backward"):
        loss.backward()
    with nvtx("cebench/optimizer"):
        optimizer.step()
    return loss.detach()


def compare_tensor(actual, expected, tolerance):
    mismatch_count = 0
    max_abs = 0.0
    finite = True
    # Bound temporary comparison storage for XL's largest parameters.
    for got, ref in zip(actual.reshape(-1).split(1_048_576), expected.reshape(-1).split(1_048_576), strict=True):
        ref = ref.to(got.device)
        finite_chunk = bool((torch.isfinite(got).all() & torch.isfinite(ref).all()).item())
        finite &= finite_chunk
        if finite_chunk:
            max_abs = max(max_abs, (got - ref).abs().max().item())
        mismatch_count += (~torch.isclose(got, ref, **tolerance)).sum().item()
    return {"finite": finite, "max_abs_error": max_abs if finite else None,
            "mismatched_elements": mismatch_count}


def check_tensors(model, stage, ref_dir, recording, rank):
    rows = []
    for index, (name, param) in enumerate(model.named_parameters()):
        tensor = param.grad if stage == "gradient" else param
        if not isinstance(tensor, DTensor):
            raise RuntimeError(f"Expected a sharded {stage} DTensor for {name}")
        # All ranks participate. Only rank 0 retains the reconstructed value.
        full = tensor.detach().full_tensor()
        if rank == 0:
            path = ref_dir / f"{stage}-{index:03}.pt"
            row = {"name": name, "shape": list(full.shape), "elements": full.numel()}
            try:
                if recording:
                    row["finite"] = bool(torch.isfinite(full).all().item())
                    torch.save({"name": name, "value": full.cpu()}, path)
                else:
                    ref = torch.load(path, map_location="cpu", weights_only=True)
                    if name != ref["name"] or full.shape != ref["value"].shape:
                        raise ValueError("reference name or shape differs")
                    row.update(compare_tensor(full, ref["value"], TOLERANCES[stage]))
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                # Complete the same sequence of collectives on both ranks before failing.
                row["error"] = str(exc)
            rows.append(row)
        del full
    return rows


def check(model, optimizer, inputs, targets, args, metadata, rank):
    ref_dir = args.reference
    recording = args.variant == "default" and args.forward_prefetch == "default"
    if rank == 0:
        if recording:
            ref_dir.mkdir(parents=True, exist_ok=True)
            (ref_dir / "manifest.json").unlink(missing_ok=True)
    dist.barrier()
    if not recording:
        reference_manifest = json.loads((ref_dir / "manifest.json").read_text())
        for key in ("config", "seed", "torch", "cuda", "nccl", "world_size", "workload"):
            if reference_manifest[key] != metadata[key]:
                raise RuntimeError(f"Reference mismatch for {key}: {reference_manifest[key]!r} vs {metadata[key]!r}")
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(model, inputs, targets)
    loss.backward()
    losses = [torch.empty_like(loss) for _ in range(dist.get_world_size())]
    dist.all_gather(losses, loss.detach())
    gradients = check_tensors(model, "gradient", ref_dir, recording, rank)
    optimizer.step()
    parameters = check_tensors(model, "parameter", ref_dir, recording, rank)
    if rank != 0:
        return None

    losses = torch.stack(losses)
    loss_check = {"finite": bool(losses.isfinite().all().item())}
    if not recording:
        loss_check.update(compare_tensor(losses, torch.tensor(reference_manifest["rank_losses"]), TOLERANCES["loss"]))
    rows = [loss_check, *gradients, *parameters]
    passed = all(row.get("finite", False) and not row.get("mismatched_elements") and "error" not in row for row in rows)
    result = {"rank_losses": [value if math.isfinite(value) else None for value in losses.tolist()],
              "tolerances": TOLERANCES, "loss_check": loss_check, "gradients": gradients, "parameters": parameters,
              "passed": passed, "recorded_reference": recording}
    if recording and passed:
        write_json(ref_dir / "manifest.json", metadata | {"rank_losses": result["rank_losses"]})
    return result


def mark_forward(module, label, trace):
    forward = module.forward

    @wraps(forward)
    def marked(*args, **kwargs):
        with torch.cuda.nvtx.range(label) if trace.active else nullcontext():
            return forward(*args, **kwargs)

    module.forward = marked


class GatherTrace:
    """Trace-only probes around the original FSDP2 calls and stream dependencies."""

    def __init__(self, model):
        self.active = False
        self.pending = {}
        self.waits = []
        for index, layer in enumerate(model.layers):
            name = f"{index:02}"
            prefix = f"cebench/layer={name}"
            self.mark_group(layer, name)
            mark_forward(layer, f"{prefix}/forward", self)
            for projection in ("q_proj", "k_proj", "v_proj", "output_proj"):
                mark_forward(getattr(layer.attn, projection), f"{prefix}/gemm={projection}", self)
            for projection in ("w1", "w2", "w3"):
                mark_forward(getattr(layer.ffn, projection), f"{prefix}/gemm={projection}", self)
        self.mark_group(model, "root")

    def mark_group(self, module, name):
        groups = module._get_fsdp_state()._fsdp_param_groups
        if len(groups) != 1:
            raise RuntimeError(f"Expected one PyTorch FSDP parameter group for layer {name}")
        group, = groups
        unshard = group.unshard
        wait_for_unshard = group.wait_for_unshard

        def traced_unshard(async_op=False):
            if not self.active or group._all_gather_result is not None or group.is_unsharded:
                return unshard(async_op)
            phase = {"FORWARD": "forward", "PRE_BACKWARD": "backward"}[group._training_state.name]
            with torch.cuda.nvtx.range(f"cebench/layer={name}/phase={phase}/unshard"):
                output = unshard(async_op)
                result = group._all_gather_result
                if result is not None:
                    if result.all_gather_work is not None or result.all_gather_event is None:
                        raise RuntimeError("Trace requires the default event-based FSDP2 unshard")
                    _, stream = group.comm_ctx.get_all_gather_streams(async_op, group._training_state)
                    # This timestamp marker follows FSDP2's own untimed dependency event.
                    # The original event and wait stay intact.
                    ready = torch.cuda.Event(enable_timing=True)
                    ready.record(stream)
                    tensor = result.all_gather_output
                    self.pending[name] = (phase, ready, tensor.numel() * tensor.element_size())
            return output

        def traced_wait():
            if not self.active or group._all_gather_result is None:
                return wait_for_unshard()
            phase, gather_ready, payload_bytes = self.pending.pop(name)
            with torch.cuda.nvtx.range(f"cebench/layer={name}/phase={phase}/wait"):
                compute_ready = torch.cuda.Event(enable_timing=True)
                compute_ready.record()
                output = wait_for_unshard()
                copy_out_done = torch.cuda.Event(enable_timing=True)
                copy_out_done.record()
            self.waits.append((name, phase, payload_bytes, compute_ready, gather_ready, copy_out_done))
            return output

        group.unshard = traced_unshard
        group.wait_for_unshard = traced_wait

    def start(self):
        self.active = True
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor.record()

    def finish(self, step):
        self.active = False
        if self.pending:
            raise RuntimeError(f"Gather events without a corresponding consumer wait: {list(self.pending)}")
        rows = []
        for name, phase, nbytes, ready, gather, done in self.waits:
            ready_ms, gather_ms, done_ms = (self.anchor.elapsed_time(event) for event in (ready, gather, done))
            rows.append({"step": step, "layer": name, "phase": phase, "payload_bytes": nbytes,
                         "compute_ready_ms": ready_ms, "gather_ready_ms": gather_ms, "copy_out_done_ms": done_ms,
                         "exposed_wait_ms": max(0.0, gather_ms - ready_ms)})
        self.waits.clear()
        return rows


def sample_device_memory(step, device):
    """Best-effort NVML samples in a separate pass, without profiling or timed samples."""
    interval = 0.010
    stop = threading.Event()
    values = []
    failures = []

    def read_memory():
        try:
            values.append(torch.cuda.device_memory_used(device))
        except Exception as exc:
            failures.append(str(exc))
            return False
        return True

    def sample():
        while not stop.is_set() and read_memory():
            stop.wait(interval)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        for _ in range(3):
            step()
        torch.cuda.synchronize(device)
    finally:
        stop.set()
        worker.join()
    return {"observed_max_device_bytes": max(values, default=None), "samples": len(values),
            "interval_ms": interval * 1000, "source": "NVML device-wide, separate three-step pass",
            "error": failures[0] if failures else None}


def run_steps(model, optimizer, inputs, targets, args, rank, device):
    profiling = args.mode == "trace"
    trace = GatherTrace(model) if profiling else None
    for _ in range(args.warmup):
        measure_training_step(model, optimizer, inputs, targets)
    torch.cuda.synchronize(device)
    waits = []
    samples = []
    for step in range(args.steps):
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        capture = torch.cuda.nvtx.range("fsdp2.measurement") if profiling else nullcontext()
        label = torch.cuda.nvtx.range(f"cebench/rank={rank}/step={step}") if profiling else nullcontext()
        with capture, label:
            if trace is not None:
                trace.start()
            start = time.perf_counter()
            loss = measure_training_step(model, optimizer, inputs, targets, annotate=profiling)
            torch.cuda.synchronize(device)
            duration_ms = (time.perf_counter() - start) * 1000
        if trace is not None:
            waits.extend(trace.finish(step))
        samples.append({"step": step, "duration_ms": duration_ms, "loss": loss.item(),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    result = {"samples": samples, "warmup": args.warmup, "profiled": profiling}
    if profiling:
        result["waits"] = waits
        result["wait_measurement"] = (
            "Per-rank trace-only estimate from two timed CUDA events. The compute marker is enqueued "
            "immediately before the native wait, and the gather marker immediately after the gather "
            "dependency event. The positive timestamp difference measures exposed readiness latency "
            "up to probe overhead and excludes subsequent weight copy-out."
        )
    else:
        dist.barrier()
        torch.cuda.synchronize(device)
        result["device_memory"] = sample_device_memory(
            lambda: measure_training_step(model, optimizer, inputs, targets), device
        )
    return result


def summarize_steps(results):
    samples = [max(rank_steps, key=lambda row: row["duration_ms"])
               for rank_steps in zip(*(result["samples"] for result in results), strict=True)]
    durations = [row["duration_ms"] for row in samples]
    return {"slowest_rank_per_step": samples, "mean_ms": statistics.mean(durations),
            "median_ms": statistics.median(durations), "stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "min_ms": min(durations), "max_ms": max(durations),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for result in results for row in result["samples"]),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for result in results for row in result["samples"])}


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Use torchrun with two local GPU ranks")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    cta_policy = "ZERO" if args.variant == "zero" else "DEFAULT"
    os.environ["NCCL_CTA_POLICY"] = cta_policy
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        model, optimizer, inputs, targets, config = make_workload(args, device, rank)
        metadata = {"rank": rank, "world_size": dist.get_world_size(), "gpu": torch.cuda.get_device_name(device),
                    "torch": str(torch.__version__), "cuda": torch.version.cuda, "nccl": list(torch.cuda.nccl.version()),
                    "mode": args.mode, "variant": args.variant, "workload": args.workload, "seed": args.seed,
                    "forward_prefetch": args.forward_prefetch, "nccl_cta_policy": cta_policy, "config": config}
        if args.mode == "check":
            outcome = check(model, optimizer, inputs, targets, args, metadata, rank)
            passed = torch.tensor(bool(outcome and outcome["passed"]), dtype=torch.int32, device=device)
            dist.broadcast(passed, src=0)
            write_json(args.output / f"rank{rank}.json", metadata | {"check": outcome, "passed": bool(passed.item())})
            if not passed.item():
                raise RuntimeError(f"Numerical parity failed. See {args.output / 'rank0.json'}")
        else:
            result = run_steps(model, optimizer, inputs, targets, args, rank, device)
            for row in result["samples"]:
                row["rank"] = rank
            write_json(args.output / f"rank{rank}.json", metadata | result)
            if args.mode == "bench":
                results = [None] * dist.get_world_size()
                dist.all_gather_object(results, result)
                if rank == 0:
                    write_json(args.output / "summary.json", metadata | summarize_steps(results))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
