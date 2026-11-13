"""Compare a staff XL block all-gather and a matching projection in isolation and overlap."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllGather, SymmMemAllGather


D_MODEL = 2560
D_FF = 10240
LOCAL_BATCH = 2
SEQUENCE = 512
# Four attention projections, three SwiGLU projections and two RMSNorm weights.
BLOCK_ELEMENTS = 4 * D_MODEL**2 + 3 * D_MODEL * D_FF + 2 * D_MODEL
PROJECTIONS = {"q": D_MODEL, "w1": D_FF}
CASES = ("gather_only", "gemm_only", "overlap_gather_first", "overlap_gemm_first")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("default", "symm", "zero"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--projection", choices=PROJECTIONS, default="q")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1:
        parser.error("--warmup and --steps must be positive")
    return args


def pair(stream, anchor, operation, label, annotate):
    with torch.cuda.stream(stream):
        stream.wait_event(anchor)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.nvtx.range(label) if annotate else nullcontext():
            start.record()
            output = operation()
            end.record()
    return start, end, output


def sample(case, index, rank, device, streams, ops, annotate):
    dist.barrier()
    torch.cuda.synchronize(device)
    order = {
        "gather_only": ("gather",),
        "gemm_only": ("gemm",),
        "overlap_gather_first": ("gather", "gemm"),
        "overlap_gemm_first": ("gemm", "gather"),
    }[case]
    anchor = torch.cuda.Event(enable_timing=True)
    capture = torch.cuda.nvtx.range("fsdp2.microbench") if annotate else nullcontext()
    scope = torch.cuda.nvtx.range(f"cemicro/rank={rank}/case={case}/step={index}") if annotate else nullcontext()
    with capture, scope:
        anchor.record()
        events = {op: pair(streams[op], anchor, ops[op], f"cemicro/{op}", annotate) for op in order}
        torch.cuda.synchronize(device)
    result = {"step": index, "case": case}
    for op, (start, end, _) in events.items():
        result[f"{op}_start_ms"] = anchor.elapsed_time(start)
        result[f"{op}_end_ms"] = anchor.elapsed_time(end)
        result[f"{op}_ms"] = start.elapsed_time(end)
    if len(order) == 2:
        start = max(result["gather_start_ms"], result["gemm_start_ms"])
        end = min(result["gather_end_ms"], result["gemm_end_ms"])
        result["overlap_ms"] = max(0, end - start)
        result["gemm_overlap_fraction"] = result["overlap_ms"] / result["gemm_ms"] if result["gemm_ms"] else 0
    return result


def summarize(samples):
    summary = {}
    for case in CASES:
        rows = [row for row in samples if row["case"] == case]
        summary[case] = {
            key: statistics.median(row[key] for row in rows)
            for key in ("gather_ms", "gemm_ms", "overlap_ms", "gemm_overlap_fraction") if key in rows[0]
        }
        if case.startswith("overlap"):
            summary[case]["positive_overlap_samples"] = sum(row["overlap_ms"] > 0 for row in rows)
    return summary


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Use torchrun with two local GPU ranks")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    policy = "ZERO" if args.variant == "zero" else "DEFAULT"
    os.environ["NCCL_CTA_POLICY"] = policy
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        group = dist.group.WORLD
        dtype = torch.float32
        comm = DefaultAllGather() if args.variant == "default" else SymmMemAllGather(group)
        gathered = comm.allocate((BLOCK_ELEMENTS,), dtype=dtype, device=device)
        shard = gathered.chunk(world_size)[rank]
        shard.fill_(rank + 1)

        generator = torch.Generator(device=device).manual_seed(2026 + rank)
        tokens = torch.randn((LOCAL_BATCH, SEQUENCE, D_MODEL), device=device, dtype=dtype, generator=generator)
        weights = torch.randn((PROJECTIONS[args.projection], D_MODEL), device=device, dtype=dtype, generator=generator)
        streams = {"gather": torch.cuda.Stream(priority=-1), "gemm": torch.cuda.Stream()}
        ops = {"gather": lambda: comm(gathered, shard, group, async_op=False),
               "gemm": lambda: torch.einsum("...i,oi->...o", tokens, weights)}
        torch.cuda.synchronize(device)
        for index in range(args.warmup):
            for case in CASES:
                sample(case, index, rank, device, streams, ops, False)
        for source, values in enumerate(gathered.chunk(world_size)):
            if not values.eq(source + 1).all().item():
                raise RuntimeError(f"Incorrect contents for all-gather input from rank {source}")
        samples = []
        for index in range(args.steps):
            # Rotate positions so each workload samples both earlier and later execution.
            shift = index % len(CASES)
            for case in CASES[shift:] + CASES[:shift]:
                samples.append(sample(case, index, rank, device, streams, ops, args.trace))
        data = {
            "rank": rank, "gpu": torch.cuda.get_device_name(device), "variant": args.variant,
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "nccl": list(torch.cuda.nccl.version()),
            "nccl_cta_policy": policy, "gather_result_bytes": gathered.nbytes, "gather_local_shard_bytes": shard.nbytes,
            "projection": args.projection, "input_shape": list(tokens.shape), "weight_shape": list(weights.shape),
            "dtype": str(dtype), "tf32": False, "warmup": args.warmup, "profiled": args.trace,
            "samples": samples, "median": summarize(samples),
            "timing_note": "CUDA events bound stream execution. Use Nsight GPU activities to verify the active collective "
                           "and GEMM intersection. Gather input is packed before timing, and no weight copy-out is included.",
        }
        path = args.output / f"rank{rank}.json"
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
