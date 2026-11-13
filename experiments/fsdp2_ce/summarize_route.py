"""Audit the fixed-staging SM-versus-copy-engine all-gather control."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from analyze_trace import analyze, compare, distribution as trace_distribution
from check_smoke import nonempty, successful, topology_rows
from summarize import (
    GIB, WARMUP, XL_CONFIG, allocator_peaks, bench_case, check_case, distribution,
    is_finite, load_case, read_json, require, validate_runtime,
)
from summarize_traces import RANKS, as_markdown as trace_markdown, summarize_comparison, summarize_variant


THRESHOLD = "1073741824"
VARIANTS = ("symm", "zero")
ROTATIONS = (VARIANTS, VARIANTS[::-1])
EXPECTED_ROUTES = {"symm": "sm_kernel", "zero": "copy_engine"}


def verify_cases(root, run):
    labels = [f"check-{variant}" for variant in ("default", *VARIANTS)]
    labels += [f"bench-{rotation}-{variant}" for rotation, order in enumerate(ROTATIONS) for variant in order]
    labels += [f"trace-{variant}" for variant in VARIANTS]
    cases = run["cases"]
    require(run["stage"] == "route" and run["status"] == "ok" and
            [case["label"] for case in cases] == labels and read_json(root / "progress.json") == cases,
            "Expected all nine cases in the original route-stage order")
    for case in cases:
        label = case["label"]
        require(case == read_json(root / label / "status.json"), f"{label}: launcher and case records disagree")
        require(case["status"] == "ok", f"{label}: failed case")
        successful(case["worker"], f"{label}/worker")
        require(case["worker"]["environment"].get("NCCL_SYM_CE_THRESHOLD") == THRESHOLD,
                f"{label}: missing fixed NCCL_SYM_CE_THRESHOLD")
        command = case["worker"]["command"]
        for key, value in (("workload", "xl"), ("variant", case["variant"]),
                           ("mode", case["mode"]), ("forward-prefetch", "default")):
            flag = f"--{key}"
            require(flag in command and command[command.index(flag) + 1] == value,
                    f"{label}: worker command has unexpected {flag}")


def source_hashes(root):
    hashes = read_json(root / "source-hashes.json")
    for name in ("bench.py", "modal_run.py", "analyze_trace.py"):
        require(f"experiments/fsdp2_ce/{name}" in hashes, f"Missing source provenance for {name}")
    for name, digest in hashes.items():
        source = nonempty(root / "source" / name)
        require(hashlib.sha256(source.read_bytes()).hexdigest() == digest, f"Source hash mismatch for {name}")
    return hashes


def topology(root):
    query = nonempty(root / "nvidia-smi.txt").read_text()
    gpu_uuids = re.findall(r"(?m)^\s*GPU UUID\s*:\s*(GPU-[a-f0-9-]+)", query, re.IGNORECASE)
    driver = re.search(r"(?m)^\s*Driver Version\s*:\s*(\S+)", query)
    require(len(set(gpu_uuids)) == 2 and len(gpu_uuids) == 2 and driver is not None,
            "Missing distinct physical GPU identities or driver")
    paths = []
    p2p = topology_rows(root / "p2p-nvlink.txt")
    if all(p2p.get(f"GPU{rank}", ["", ""])[1 - rank] == "OK" for rank in range(2)):
        paths.append("p2p-nvlink.txt")
    for variant in VARIANTS:
        log = nonempty(root / f"trace-{variant}" / "worker.log")
        text = log.read_text()
        ranks = re.findall(r"\brank ([01]) nRanks 2 nNodes 1 localRanks 2 localRank [01]\b", text)
        require(set(ranks) == set(RANKS), f"{log}: both NCCL ranks must be local")
        if re.search(r"GPU/[a-f0-9]+-[a-f0-9]+ \(\d+/[\d.]+/NVL\)", text, re.IGNORECASE):
            paths.append(str(log.relative_to(root)))
    require(paths, "Missing pairwise NVLink evidence")
    return {"gpu_uuids": gpu_uuids, "driver_version": driver[1], "nvlink_evidence": paths}


def measure_steps(root, runtime, config):
    rotations = []
    for index, order in enumerate(ROTATIONS):
        cases = {variant: bench_case(root, f"bench-{index}-{variant}", variant, runtime, config)
                 for variant in order}
        before = cases["symm"]["timing"]["median_ms"]
        after = cases["zero"]["timing"]["median_ms"]
        rotations.append({"rotation": index, "execution_order": order, "cases": cases,
                          "median_delta_ms": after - before, "median_ratio": after / before})
    variants = {}
    for variant in VARIANTS:
        cases = [row["cases"][variant] for row in rotations]
        memory = [row for case in cases for row in case["memory_by_rank"]]
        observed = [row["nvml"]["observed_max_device_bytes"] for row in memory]
        variants[variant] = {
            "timing": distribution([row["max_rank_ms"] for case in cases for row in case["steps"]]),
            **allocator_peaks(memory),
            "observed_max_device_bytes": max((value for value in observed if value is not None), default=None),
            "nvml_rank_runs_sampled": sum(value is not None for value in observed),
            "nvml_rank_runs_expected": len(observed),
        }
    return rotations, variants


def verify_trace(root, variant, runtime, config):
    label = f"trace-{variant}"
    folder = root / label
    status = read_json(folder / "status.json")
    for stage in ("export", "analysis"):
        successful(status[stage], f"{label}/{stage}")
    nonempty(folder / "profile.nsys-rep")
    raw = nonempty(folder / "profile.sqlite")
    rows = load_case(root, label, variant, "trace", runtime, config)
    trace = analyze(raw, folder, expected_ranks=2)
    require(trace["two_rank_gpu_coverage"], f"{label}: both ranks must have distinct GPU activity")
    blocks = {f"{index:02d}" for index in range(config["num_layers"])}
    expected = {(layer, "forward") for layer in blocks | {"root"}} | {(layer, "backward") for layer in blocks}
    for rank, row in enumerate(rows):
        samples = row["samples"]
        evidence = trace["ranks"][str(rank)]
        require(row["profiled"] is True and row["warmup"] == WARMUP and len(samples) == 2 and
                all(sample["step"] == step and sample["rank"] == rank and is_finite(sample["duration_ms"])
                    and sample["duration_ms"] > 0 and is_finite(sample["loss"]) for step, sample in enumerate(samples)),
                f"{label}: missing profiled samples for rank {rank}")
        require(evidence["devices"] == [rank] and evidence["unmatched_wait_records"] == 0 and
                [step["step"] for step in evidence["steps"]] == [0, 1], f"{label}: incomplete rank {rank} trace")
        for step in evidence["steps"]:
            layers = step["layers"]
            require(step["devices"] == [rank] and step["gemms"] and len(layers) == len(expected) and
                    {(layer["layer"], layer["phase"]) for layer in layers} == expected and
                    all(len(layer["gathers"]) == 1 and layer["measured_wait"] is not None for layer in layers),
                    f"{label}: incomplete layer/gather/wait scope for rank {rank}, step {step['step']}")
        # The shared trace summarizer consumes JSON exports with string step keys.
        evidence["profiled_step_duration_ms"] = {
            str(step): duration for step, duration in evidence["profiled_step_duration_ms"].items()
        }
    return trace


def trace_metrics(trace):
    result = {}
    for rank, row in trace["ranks"].items():
        steps = []
        for step in row["steps"]:
            phases = {}
            for phase in ("forward", "backward"):
                layers = [layer for layer in step["layers"] if layer["phase"] == phase]
                gathers = [layer["gathers"][0] for layer in layers]
                durations = [gather["gpu_active_ns"] / 1e6 for gather in gathers if gather["gpu_active_ns"] > 0]
                phases[phase] = {
                    "gather_active_ms": trace_distribution(durations),
                    "exposed_wait_ms": trace_distribution([layer["measured_wait"]["exposed_wait_ms"] for layer in layers]),
                }
            steps.append({"step": step["step"], "phases": phases})
        result[rank] = steps
    return result


def largest_ce_waits(trace):
    candidates = []
    for rank, data in trace["ranks"].items():
        for step in data["steps"]:
            for layer in step["layers"]:
                wait = layer["measured_wait"]["exposed_wait_ms"]
                gather = layer["gathers"][0]
                if wait > 0 and gather["route"] == "copy_engine":
                    candidates.append((wait, rank, step["step"], layer["layer"], layer["phase"], gather))
    result = []
    for wait, rank, step, layer, phase, gather in sorted(candidates, key=lambda row: row[0], reverse=True)[:4]:
        gpu = gather["gpu_execution"]
        row = {"rank": rank, "step": step, "layer": layer, "phase": phase, "exposed_wait_ms": wait,
               "gpu_transfer_start_ns": gpu["start_ns"], "gpu_active_ms": gather["gpu_active_ns"] / 1e6}
        apis = gather["nccl_allgather_api"]
        if len(apis) == 1 and len(api_ids := apis[0]["payload"].get("apiid", [])) == 1:
            api_id = api_ids[0]
            peer_rank = RANKS[1 - int(rank)]
            peer_apis = [api for peer_layer in trace["ranks"][peer_rank]["steps"][step]["layers"]
                         if (peer_layer["layer"], peer_layer["phase"]) == (layer, phase)
                         for peer_gather in peer_layer["gathers"] for api in peer_gather["nccl_allgather_api"]
                         if api_id in api["payload"].get("apiid", [])]
            if len(peer_apis) == 1:
                issue = apis[0]["start"]
                peer_issue = peer_apis[0]["start"]
                row.update(api_id=api_id, api_issue_ns=issue, peer_api_issue_ns=peer_issue,
                           api_to_transfer_ms=(gpu["start_ns"] - issue) / 1e6,
                           peer_issue_skew_ms=(peer_issue - issue) / 1e6,
                           peer_api_to_local_transfer_ms=(gpu["start_ns"] - peer_issue) / 1e6)
        result.append(row)
    return result


def summarize(root):
    run = read_json(root / "summary.json")
    verify_cases(root, run)
    runtime = read_json(root / "runtime.json")
    gpu_model, _ = validate_runtime(runtime)
    require(run.get("gpu_model", gpu_model) == gpu_model, "Declared GPU model disagrees with runtime")
    hashes = source_hashes(root)
    config = read_json(root / "check-default" / "rank0.json")["config"]
    require(all(config[key] == value for key, value in XL_CONFIG.items()) and
            re.fullmatch("[0-9a-f]{64}", config["input_sha256"]), "Unexpected XL workload configuration")
    reference, reference_shapes = check_case(root, "check-default", "default", runtime, config)
    parity = {"default": reference}
    for variant in VARIANTS:
        parity[variant], _ = check_case(root, f"check-{variant}", variant, runtime, config,
                                       reference_shapes=reference_shapes)
    rotations, variants = measure_steps(root, runtime, config)
    raw = {variant: verify_trace(root, variant, runtime, config) for variant in VARIANTS}
    traces = {variant: summarize_variant(trace, variant) for variant, trace in raw.items()}
    route_confirmed = all(set(traces[variant]["ranks"][rank]["gather_routes"]) == {expected_route}
                          for variant, expected_route in EXPECTED_ROUTES.items() for rank in RANKS)
    paired = compare(raw["zero"], raw["symm"])
    matched_scopes = Counter((pair["rank"], pair["step"]) for pair in paired["pairs"])
    require(all(matched_scopes[rank, step] for rank in RANKS for step in range(2)),
            "No matched projection GEMMs in one or more rank/step scopes")
    return {
        "run_id": run["run_id"], "gpu_model": gpu_model, "runtime": runtime, "topology": topology(root),
        "source_hashes": hashes, "config": config, "parity": parity,
        "environment": {"NCCL_SYM_CE_THRESHOLD": THRESHOLD},
        "route_confirmed": route_confirmed, "expected_routes": EXPECTED_ROUTES,
        "rotations": rotations, "variants": variants, "traces": traces,
        "trace_distributions": {variant: trace_metrics(trace) for variant, trace in raw.items()},
        "largest_ce_waits": largest_ce_waits(raw["zero"]),
        "comparison": summarize_comparison(paired, traces["zero"], traces["symm"]),
        "notes": [
            "Both timing variants use symmetric memory and the same CE threshold. The NCCL CTA policy differs.",
            "Complete-step timing uses the maximum of the two ranks per step without profiling. NVML maxima are sampled in separate passes.",
            "Trace timing is diagnostic. Exposed waits are device event gaps before parameter copy-out, and GEMM duration "
            "differences do not by themselves identify specific resource contention.",
        ],
    }


def as_markdown(result):
    lines = [
        f"# {result['run_id']} · SM versus copy-engine control", "",
        f"Hardware is two {result['gpu_model']} GPUs. Both policies use symmetric memory and "
        f"`NCCL_SYM_CE_THRESHOLD={result['environment']['NCCL_SYM_CE_THRESHOLD']}`.",
        "",
        "Raw trace confirms the expected SM and copy-engine routes on both ranks." if result["route_confirmed"] else
        "Raw trace does not establish an SM-versus-copy-engine comparison.",
        "",
        "| Policy | Steps | Median (ms) | P95 (ms) | Peak allocated (GiB) | Peak reserved (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in result["variants"].items():
        timing = row["timing"]
        lines.append(f"| {variant} | {timing['count']} | {timing['median_ms']:.3f} | {timing['p95_ms']:.3f} | "
                     f"{row['peak_allocated_bytes'] / GIB:.3f} | {row['peak_reserved_bytes'] / GIB:.3f} |")
    lines += ["", "| Rotation | Execution order | Symmetric median (ms) | Zero-CTA median (ms) | Δ (ms) | Ratio |",
              "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in result["rotations"]:
        timing = row["cases"]
        lines.append(f"| {row['rotation'] + 1} | {' → '.join(row['execution_order'])} | "
                     f"{timing['symm']['timing']['median_ms']:.3f} | {timing['zero']['timing']['median_ms']:.3f} | "
                     f"{row['median_delta_ms']:+.3f} | {row['median_ratio']:.4f}× |")
    lines += ["", "Both policies passed rank loss, full gradient, and post-AdamW parameter checks "
              "against the default FSDP2 reference.", ""]
    waits = [row for row in result["largest_ce_waits"] if "api_id" in row]
    if waits:
        lines += ["## Largest CE waits with matching peer NCCL API IDs", "",
                  "| Rank | Step | Layer | Exposed wait (ms) | API to GPU copy (ms) | Peer API skew (ms) | GPU copy active (ms) |",
                  "| ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
        for row in waits:
            lines.append(f"| {row['rank']} | {row['step']} | {row['phase']} {row['layer']} | "
                         f"{row['exposed_wait_ms']:.3f} | {row['api_to_transfer_ms']:.3f} | "
                         f"{row['peer_issue_skew_ms']:+.3f} | {row['gpu_active_ms']:.3f} |")
        lines += ["", "Peer API skew is peer minus local entry on the shared trace clock. "
                  "It does not identify why one rank entered later.", ""]
    rendered_trace = trace_markdown({
        "run_id": result["run_id"], "variants": result["traces"],
        "comparisons": {"zero_vs_symm": result["comparison"]}, "notes": result["notes"],
    })
    return "\n".join(lines) + "\n" + rendered_trace.replace("# ", "## ", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.run_dir)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n" if args.format == "json" else as_markdown(result)
        if args.output:
            args.output.write_text(content)
        else:
            print(content, end="")
    except (IndexError, KeyError, OSError, TypeError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"Invalid route-run evidence: {error}\n")
    if not result["route_confirmed"]:
        parser.exit(2, "Expected SM vs copy-engine routes were not proven on both ranks\n")


if __name__ == "__main__":
    main()
