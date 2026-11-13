"""Check the same-size all-gather / projection microbenchmark against a Nsight SQLite export."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from analyze_trace import GEMM, Trace, cpu_ref, distribution, gpu_ref, union_duration

SCOPE = re.compile(r"cemicro/rank=(\d+)/case=([^/]+)/step=(\d+)$")
CASE_OPS = {
    "gather_only": ("gather",),
    "gemm_only": ("gemm",),
    "overlap_gather_first": ("gather", "gemm"),
    "overlap_gemm_first": ("gemm", "gather"),
}


def read_step(trace, scope, operations):
    issues = []
    marks = {}
    for row in trace.nvtx_processes.within(scope["pid"], scope["start"], scope["end"]):
        if row["globalTid"] != scope["globalTid"] or row["end"] > scope["end"]:
            continue
        for name in ("gather", "gemm"):
            if row["name"] != f"cemicro/{name}":
                continue
            if name in marks:
                issues.append(f"duplicate {name} scope")
            marks[name] = row
    if set(marks) != set(operations):
        issues.append(f"expected {', '.join(operations)} scopes; captured {', '.join(marks) or 'none'}")
    elif tuple(sorted(marks, key=lambda op: marks[op]["start"])) != operations:
        issues.append("operation scopes are out of order for this case")

    gather = trace.gather(marks["gather"]) if "gather" in marks else None
    comm_intervals = gather.pop("intervals") if gather else []
    if "gather" in operations and not comm_intervals:
        issues.append("no correlated all-gather CUDA copy or NCCL GPU kernel")

    gemms = {}
    if "gemm" in marks:
        for _, gpu in trace.activities(marks["gemm"]):
            if gpu.get("shortName") and GEMM.search(gpu["name"]):
                key = (gpu["correlationId"], gpu["streamId"], gpu["start"], gpu["end"])
                gemms[key] = gpu
    gemm_intervals = [(row["start"], row["end"]) for row in gemms.values()]
    if "gemm" in operations and not gemm_intervals:
        issues.append("no correlated projection GEMM GPU kernel")

    devices = {row["deviceId"] for row in gemms.values()}
    if gather:
        devices.update(row["deviceId"] for row in gather["sm_kernels"] + gather["copy_ops"])
    if len(devices) != 1:
        issues.append(f"expected one observed GPU for this rank; captured devices {sorted(devices)}")
    if comm_intervals and gemm_intervals:
        gather_streams = {row["streamId"] for row in gather["sm_kernels"] + gather["copy_ops"]}
        gemm_streams = {row["streamId"] for row in gemms.values()}
        if gather_streams & gemm_streams:
            issues.append("gather and GEMM activity share a CUDA stream in an overlap case")
    if any(start < scope["start"] or end > scope["end"] or start >= end
           for start, end in comm_intervals + gemm_intervals):
        issues.append("GPU activity falls outside the synchronized sample scope")

    if comm_intervals and gemm_intervals:
        overlap_ns = union_duration(
            (max(comm_start, gemm_start), min(comm_end, gemm_end))
            for comm_start, comm_end in comm_intervals
            for gemm_start, gemm_end in gemm_intervals
            if comm_start < gemm_end and gemm_start < comm_end
        )
    else:
        overlap_ns = None
    gemm_active = union_duration(gemm_intervals) if gemm_intervals else None
    return {
        "cpu_marker": cpu_ref(scope),
        "devices": sorted(devices),
        "gather": gather,
        "gather_gpu_active_ns": gather["gpu_active_ns"] if comm_intervals else None,
        "gemm_kernels": [
            {**gpu_ref(row), "duration_ns": row["end"] - row["start"]}
            for row in sorted(gemms.values(), key=lambda row: row["start"])
        ],
        "gemm_gpu_active_ns": gemm_active,
        "gpu_intersection_ns": overlap_ns,
        "gemm_intersection_fraction": overlap_ns / gemm_active if overlap_ns is not None and gemm_active else None,
        "issues": issues,
    }


def summarize_case(samples):
    active_fields = (
        "gather_gpu_active_ns", "gemm_gpu_active_ns", "gpu_intersection_ns", "gemm_intersection_fraction",
    )
    return {
        "route_counts": dict(Counter(row["gather"]["route"] for row in samples if row["gather"])),
        "positive_intersection_samples": sum((row["gpu_intersection_ns"] or 0) > 0 for row in samples),
        **{field: distribution([row[field] for row in samples if row[field] is not None]) for field in active_fields},
        "samples": samples,
    }


def analyze(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        trace = Trace(db)
    rank_samples = {}
    issues = []
    for row in trace.nvtx:
        match = SCOPE.fullmatch(row["name"])
        if match is None:
            continue
        rank, case, step = match.groups()
        step = int(step)
        if rank not in {"0", "1"} or case not in CASE_OPS:
            issues.append(f"unexpected sample marker {row['name']}")
            continue
        cases = rank_samples.setdefault(rank, {name: {} for name in CASE_OPS})
        if step in cases[case]:
            issues.append(f"duplicate sample marker {row['name']}")
            continue
        sample = read_step(trace, row, CASE_OPS[case])
        issues.extend(f"{row['name']}: {issue}" for issue in sample["issues"])
        cases[case][step] = {**sample, "step": step, "process_id": row["pid"]}

    ranks = {}
    coverage = {}
    for rank in ("0", "1"):
        cases = rank_samples.get(rank, {})
        report = {}
        pids = set()
        devices = set()
        for name in CASE_OPS:
            steps = cases.get(name, {})
            if not steps:
                issues.append(f"rank {rank} has no samples for {name}")
            coverage[rank, name] = sorted(steps)
            samples = [steps[step] for step in sorted(steps)]
            pids.update(row["process_id"] for row in samples)
            devices.update(device for row in samples for device in row["devices"])
            report[name] = summarize_case(samples)
        if len(devices) != 1 or len(pids) != 1:
            issues.append(f"rank {rank} lacks a unique process/device in captured samples")
        ranks[rank] = {"processes": sorted(pids), "devices": sorted(devices), "cases": report}

    expected_steps = list(range(len(coverage["0", "gather_only"])))
    if not expected_steps or any(steps != expected_steps for steps in coverage.values()):
        issues.append("all four cases must have identical contiguous sample indices on both ranks")
    for field in ("processes", "devices"):
        if set(ranks["0"][field]) & set(ranks["1"][field]):
            issues.append(f"rank markers share {field}")

    gather_samples = [
        row for rank in ranks.values() for case, report in rank["cases"].items()
        if "gather" in CASE_OPS[case] for row in report["samples"]
    ]
    return {
        "sqlite": str(path), "two_rank_case_coverage": not issues, "ranks": ranks,
        "copy_engine_confirmed_all_gathers": not issues and all(
            row["gather"]["route"] == "copy_engine" for row in gather_samples
        ),
        "issues": issues,
        "notes": [
            "Active GPU intervals use correlated collective copies or NCCL kernels and named projection GEMM kernels.",
            "Gather GPU-active time excludes gaps, synchronization flags and post-transfer dependencies. "
            "Use the worker's CUDA event measurements for stream elapsed time.",
            "An overlap case can have zero observed intersection when one operation finishes before the other starts.",
            "Profiled kernel durations diagnose the isolated workload and do not measure full training step time.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ce", action="store_true", help="Require observed CE AllGather in every gather sample")
    args = parser.parse_args()
    try:
        result = analyze(args.sqlite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.exit(2, f"Invalid microbenchmark trace: {error}\n")
    if not result["two_rank_case_coverage"]:
        parser.exit(2, f"Incomplete microbenchmark GPU evidence: {result['issues'][0]}\n")
    if args.require_ce and not result["copy_engine_confirmed_all_gathers"]:
        parser.exit(2, "One or more gather samples lack an unambiguous copy-engine route\n")


if __name__ == "__main__":
    main()
