"""Summarize XL FSDP2 traces and any selected prefetch follow-up from a full run."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from analyze_trace import compare, union_duration
from summarize import load_case

VARIANTS = ("default", "symm", "zero")
PHASES = ("forward", "backward")
RANKS = ("0", "1")
COMPARISONS = (("symm", "default"), ("zero", "symm"), ("zero", "default"))
PREFETCH_METRICS = {
    "unshard_issue_offset_ms": "CPU unshard issue offset",
    "nccl_api_issue_offset_ms": "CPU NCCL AllGather issue offset",
    "gpu_transfer_start_offset_ms": "GPU transfer start offset",
    "nccl_issue_to_transfer_start_ms": "NCCL API entry to GPU transfer start",
    "gpu_transfer_span_ms": "GPU transfer span",
    "gpu_transfer_active_ms": "GPU transfer active time",
    "exposed_wait_ms": "Compute stream exposed wait",
    "projection_kernel_sum_ms": "Matched projection kernel duration sum",
}
PREFETCH_NOTES = [
    "Offsets are relative to the same rank's CPU step marker in each capture. Nsight maps CPU and GPU activity "
    "to its trace timeline. A negative paired offset delta means earlier within that local profiled step. "
    "Earlier-layer work can also change that offset. The API-entry-to-transfer interval combines host work, "
    "stream scheduling, and collective readiness.",
    "GPU transfer span runs from the first correlated copy start to the last copy end. "
    "Active time excludes gaps. Both omit the subsequent NCCL completion synchronization.",
    "The event wait is measured on the device from compute-ready to gather-ready, excluding copy-out. "
    "The projection total sums same-name, same-geometry kernel pairs. It is a kernel-duration statistic.",
    "Prefetch comparisons hold symmetric memory and zero-CTA fixed. They characterize these two profiled steps per rank. "
    "Use the independent unprofiled result for end-to-end step time.",
]


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_trace(root, variant, *, label=None, prefetch="default", runtime=None, config=None):
    folder = root / (label or f"trace-{variant}")
    status = read_json(folder / "status.json")
    expected = {"label": folder.name, "variant": variant, "mode": "trace", "workload": "xl", "status": "ok"}
    require(all(status.get(key) == value for key, value in expected.items()), f"{folder.name}: unexpected case status")
    for stage in ("worker", "export", "analysis"):
        run = status[stage]
        require(run["returncode"] == 0 and run["timed_out"] is False, f"{folder.name}: failed {stage}")
    for filename in ("profile.nsys-rep", "profile.sqlite"):
        path = folder / filename
        require(path.is_file() and path.stat().st_size > 0, f"{folder.name}: missing {filename}")
    report = read_json(folder / "trace.json")
    require(report["two_rank_gpu_coverage"] and set(report["ranks"]) == set(RANKS),
            f"{folder.name}: missing two-rank trace coverage")
    if variant == "zero":
        require(report["copy_engine_confirmed_all_ranks"], f"{folder.name}: CE is unproven on one or both ranks")
    if runtime is not None:
        rows = load_case(root, folder.name, variant, "trace", runtime, config, prefetch=prefetch)
        require(all(row["profiled"] is True for row in rows), f"{folder.name}: missing profiled rank metadata")
    for rank, data in report["ranks"].items():
        require(data["workload"] == "xl" and data["variant"] == variant and data["forward_prefetch"] == prefetch,
                f"{folder.name}: unexpected configuration for rank {rank}")
        steps = data["steps"]
        require(len(steps) == 2 and [step["step"] for step in steps] == [0, 1],
                f"{folder.name}: expected two captured steps for rank {rank}")
        require(all(step["kernel_count"] > 0 for step in steps), f"{folder.name}: missing GPU kernels for rank {rank}")
    return report


def summarize_phase(step, phase, scope):
    layers = [row for row in step["layers"] if row["phase"] == phase]
    rows = [row["measured_wait"] for row in layers if row["measured_wait"] is not None]
    intervals = []
    exposed_waits = []
    for row in rows:
        begin, end, exposed = (row[key] for key in ("compute_ready_ms", "gather_ready_ms", "exposed_wait_ms"))
        require(all(finite_number(value) for value in (begin, end, exposed))
                and math.isclose(exposed, max(0.0, end - begin), rel_tol=1e-9, abs_tol=1e-9),
                f"{scope}/{phase}: inconsistent exposed wait")
        intervals.append((begin, end))
        exposed_waits.append(exposed)
    return {
        "route_counts": dict(Counter(gather["route"] for row in layers for gather in row["gathers"])),
        "layer_scopes": len(layers),
        "wait_markers": len(rows),
        "positive_wait_markers": sum(wait > 0 for wait in exposed_waits),
        "unmeasured_layer_scopes": len(layers) - len(rows),
        "wait_sum_ms": math.fsum(exposed_waits) if rows else None,
        "wait_union_ms": union_duration(intervals) if rows else None,
        "native_wait_links": sum(
            wait["producer"] is not None for row in layers for wait in row["native_waits"]
        ),
        "measured_layers": sorted({row["layer"] for row in layers if row["measured_wait"] is not None}),
    }


def summarize_variant(report, variant):
    ranks = {}
    for rank, data in report["ranks"].items():
        steps = []
        for row in data["steps"]:
            step = row["step"]
            duration = data["profiled_step_duration_ms"][str(step)]
            require(finite_number(duration) and duration > 0, f"{variant} rank {rank} step {step}: invalid duration")
            phases = {phase: summarize_phase(row, phase, f"{variant} rank {rank} step {step}") for phase in PHASES}
            steps.append({
                "step": step, "profiled_duration_ms": duration, "phases": phases,
                "all_phase_wait_union_ms": row["measured_wait_union_ms"],
            })
        ranks[rank] = {
            "devices": data["devices"], "gather_routes": data["gather_routes"],
            "unmatched_wait_records": data["unmatched_wait_records"], "steps": steps,
        }
    slowest = []
    for step, (row0, row1) in enumerate(zip(*(ranks[rank]["steps"] for rank in RANKS))):
        durations = [row0["profiled_duration_ms"], row1["profiled_duration_ms"]]
        maximum = max(durations)
        slowest.append({
            "step": step, "ranks": [rank for rank, duration in enumerate(durations) if duration == maximum],
            "profiled_duration_ms": maximum,
        })
    return {
        "copy_engine_confirmed_all_ranks": report["copy_engine_confirmed_all_ranks"],
        "copy_engine_only_all_gathers": report["copy_engine_only_all_gathers"],
        "ranks": ranks, "slowest_profiled": slowest,
    }


def gemm_stats(pairs):
    return {
        "matched": len(pairs),
        **{
            key: statistics.median(pair[field] for pair in pairs) if pairs else None
            for key, field in (
                ("baseline_median_ns", "baseline_ns"), ("candidate_median_ns", "candidate_ns"),
                ("paired_delta_median_ns", "delta_ns"),
            )
        },
    }


def summarize_comparison(comparison, candidate_summary, baseline_summary):
    by_overlap = {
        "both": [], "baseline_only": [], "candidate_only": [], "neither": [],
    }
    for pair in comparison["pairs"]:
        before = pair["baseline_overlap_ns"] > 0
        after = pair["candidate_overlap_ns"] > 0
        label = {
            (True, True): "both", (True, False): "baseline_only",
            (False, True): "candidate_only", (False, False): "neither",
        }[before, after]
        by_overlap[label].append(pair)
    wait_delta_rows = []
    for rank in RANKS:
        for candidate_step, baseline_step in zip(
            candidate_summary["ranks"][rank]["steps"], baseline_summary["ranks"][rank]["steps"]
        ):
            for phase in PHASES:
                before = baseline_step["phases"][phase]
                after = candidate_step["phases"][phase]
                if before["measured_layers"] and before["measured_layers"] == after["measured_layers"]:
                    wait_delta_rows.append({
                        "rank": rank, "step": candidate_step["step"], "phase": phase,
                        "matched_wait_layers": len(before["measured_layers"]),
                        "delta_union_ms": after["wait_union_ms"] - before["wait_union_ms"],
                    })
    return {
        "gemm_matching": comparison["matching"], "all_matched_gemms": gemm_stats(comparison["pairs"]),
        "by_observed_overlap": {label: gemm_stats(pairs) for label, pairs in by_overlap.items()},
        "unmatched_baseline": comparison["baseline_only"], "unmatched_candidate": comparison["candidate_only"],
        "kernel_mismatches": comparison["kernel_mismatches"], "wait_delta_rows": wait_delta_rows,
    }


def forward_blocks(report):
    blocks = {}
    for rank, data in report["ranks"].items():
        for step in data["steps"]:
            for layer in step["layers"]:
                if layer["phase"] != "forward" or layer["layer"] in {"root", "00"}:
                    continue
                key = (rank, step["step"], layer["layer"])
                require(key not in blocks, f"Duplicate forward block {key}")
                blocks[key] = (step["cpu_marker"], layer)
    return blocks


def prefetch_block_values(marker, layer):
    gathers = layer["gathers"]
    if len(gathers) != 1 or gathers[0]["route"] != "copy_engine":
        return None
    gather = gathers[0]
    issue = gather["unshard"]["start"]
    apis = gather["nccl_allgather_api"]
    api_start = apis[0]["start"] if len(apis) == 1 else None
    transfer = gather["gpu_execution"]
    start, end = transfer["start_ns"], transfer["end_ns"]
    active = gather["gpu_active_ns"]
    require(marker["start"] <= issue < marker["end"] and marker["start"] <= start < end <= marker["end"]
            and 0 < active <= end - start, f"Inconsistent GPU/CPU trace intervals for block {layer['layer']}")
    wait = layer["measured_wait"]
    return {
        "unshard_issue_offset_ms": (issue - marker["start"]) / 1e6,
        "nccl_api_issue_offset_ms": (api_start - marker["start"]) / 1e6 if api_start is not None else None,
        "gpu_transfer_start_offset_ms": (start - marker["start"]) / 1e6,
        "nccl_issue_to_transfer_start_ms": (start - api_start) / 1e6 if api_start is not None else None,
        "gpu_transfer_span_ms": (end - start) / 1e6,
        "gpu_transfer_active_ms": active / 1e6,
        "exposed_wait_ms": wait["exposed_wait_ms"] if wait is not None else None,
    }


def prefetch_comparison(root, run, runtime, config):
    labels = ("trace-zero-default", "trace-zero-next")
    recorded = [case for case in run["cases"] if case["label"] in labels]
    require(tuple(case["label"] for case in recorded) == labels and all(case["status"] == "ok" for case in recorded),
            "Missing, failed, or misordered launcher prefetch trace cases")
    baseline = load_trace(root, "zero", label=labels[0], runtime=runtime, config=config)
    candidate = load_trace(root, "zero", label=labels[1], prefetch="next", runtime=runtime, config=config)
    summaries = {
        "default": summarize_variant(baseline, labels[0]),
        "next": summarize_variant(candidate, labels[1]),
    }
    pairs = compare(candidate, baseline)
    comparison = summarize_comparison(pairs, summaries["next"], summaries["default"])
    gemms_by_block = defaultdict(list)
    for row in pairs["pairs"]:
        gemms_by_block[row["rank"], row["step"], row["layer"]].append(row)

    before_blocks, after_blocks = forward_blocks(baseline), forward_blocks(candidate)
    rows = []
    rejected = []
    for key in sorted(before_blocks.keys() & after_blocks.keys()):
        rank, step, layer = key
        before = prefetch_block_values(*before_blocks[key])
        after = prefetch_block_values(*after_blocks[key])
        if before is None or after is None:
            rejected.append({"rank": rank, "step": step, "layer": layer})
            continue
        pairs = gemms_by_block[key]
        before["projection_kernel_sum_ms"] = sum(pair["baseline_ns"] for pair in pairs) / 1e6 if pairs else None
        after["projection_kernel_sum_ms"] = sum(pair["candidate_ns"] for pair in pairs) / 1e6 if pairs else None
        delta = {
            field: after[field] - before[field]
            for field in PREFETCH_METRICS if after[field] is not None and before[field] is not None
        }
        rows.append({
            "rank": rank, "step": step, "layer": layer, "default": before, "next": after, "delta": delta,
            "matched_projection_kernels": len(pairs),
        })

    required_coverage = {(rank, step) for rank in RANKS for step in (0, 1)}
    observed_coverage = {(row["rank"], row["step"]) for row in rows}
    require(required_coverage == observed_coverage, "No common CE-confirmed prefetchable blocks in one or more rank steps")
    medians = {}
    for field in PREFETCH_METRICS:
        paired = [row for row in rows if field in row["delta"]]
        medians[field] = {
            "matched_blocks": len(paired),
            "default_median_ms": statistics.median(row["default"][field] for row in paired) if paired else None,
            "next_median_ms": statistics.median(row["next"][field] for row in paired) if paired else None,
            "paired_delta_median_ms": statistics.median(row["delta"][field] for row in paired) if paired else None,
        }
    for field in ("exposed_wait_ms", "projection_kernel_sum_ms"):
        available = {(row["rank"], row["step"]) for row in rows if field in row["delta"]}
        require(available == required_coverage, f"No paired {field} for one or more rank steps")
    return {
        "status": "verified", "cases": dict(zip(("default", "next"), labels)), "variants": summaries,
        "matched_blocks": rows, "not_confirmed_ce_blocks": rejected,
        "unmatched_default_blocks": len(before_blocks.keys() - after_blocks.keys()),
        "unmatched_next_blocks": len(after_blocks.keys() - before_blocks.keys()),
        "block_metric_medians": medians, "gemm_and_wait_comparison": comparison, "notes": PREFETCH_NOTES,
    }


def wait_diagnostic(root, run, runtime, config):
    path = root / "wait-diagnostic.json"
    if not path.is_file():
        return {}
    diagnostic = {"path": path.name}
    try:
        diagnostic["decision"] = read_json(path)
        if diagnostic["decision"]["followup"] == "prefetch":
            diagnostic["prefetch"] = prefetch_comparison(root, run, runtime, config)
    except (KeyError, OSError, TypeError, ValueError) as error:
        diagnostic["prefetch"] = {"status": "incomplete", "error": str(error)}
    return {"wait_diagnostic": diagnostic}


def summarize(root):
    run = read_json(root / "summary.json")
    require(run["stage"] == "full", "Expected a full-stage run")
    # Older local trace bundles can contain only the per-case analyzer exports.
    # A complete downloaded run also carries runtime and workload metadata.
    runtime_path = root / "runtime.json"
    runtime = read_json(runtime_path) if runtime_path.is_file() else None
    config = read_json(root / "check-default" / "rank0.json")["config"] if runtime is not None else None
    if runtime is not None:
        names = runtime["gpu_names"]
        require(len(names) == 2 and names[0] == names[1], "Trace variants require a uniform GPU pair")
    traces = {variant: load_trace(root, variant, runtime=runtime, config=config) for variant in VARIANTS}
    variants = {variant: summarize_variant(trace, variant) for variant, trace in traces.items()}
    return {
        "run_id": run["run_id"], "variants": variants,
        "comparisons": {
            f"{candidate}_vs_{baseline}": summarize_comparison(
                compare(traces[candidate], traces[baseline]), variants[candidate], variants[baseline]
            )
            for candidate, baseline in COMPARISONS
        },
        **wait_diagnostic(root, run, runtime, config),
        "notes": [
            "All timing here is profiled. Use the separately collected unprofiled complete steps for performance claims.",
            "Exposed wait starts when the compute-ready marker reaches the GPU. Waits exclude earlier CPU issuance delay "
            "and later parameter copy-out. Per-phase union deduplicates overlapping marker intervals. A missing marker is unmeasured.",
            "GEMM overlap counts use proven all-gather activity on the same rank. Unresolved routes may hide overlaps. "
            "Matched profiled kernel duration differences do not identify resource contention without further controls.",
        ],
    }


def as_markdown(result):
    def ms(value):
        return "unmeasured" if value is None else f"{value:.3f}"

    lines = [
        f"# {result['run_id']} · Nsight trace", "",
        "| Variant | Rank | Step | Phase | Wait markers / layers | Wait sum (ms) | Wait union (ms) | All-gather routes |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for variant, report in result["variants"].items():
        for rank, data in report["ranks"].items():
            for step in data["steps"]:
                for phase, metrics in step["phases"].items():
                    routes = ", ".join(f"{route} {count}" for route, count in sorted(metrics["route_counts"].items()))
                    lines.append(
                        f"| {variant} | {rank} | {step['step']} | {phase} | "
                        f"{metrics['wait_markers']} / {metrics['layer_scopes']} | "
                        f"{ms(metrics['wait_sum_ms'])} | {ms(metrics['wait_union_ms'])} | {routes or 'none observed'} |"
                    )

    lines += ["", "| Variant | Step | Slowest profiled rank | Profiled step (ms) |",
              "| --- | ---: | ---: | ---: |"]
    for variant, report in result["variants"].items():
        for row in report["slowest_profiled"]:
            ranks = ", ".join(map(str, row["ranks"]))
            lines.append(f"| {variant} | {row['step']} | {ranks} | {ms(row['profiled_duration_ms'])} |")

    lines += ["", "| Comparison | Rank | Step | Phase | Same measured layers | Wait union Δ (ms) |",
              "| --- | ---: | ---: | --- | ---: | ---: |"]
    for name, comparison in result["comparisons"].items():
        for row in comparison["wait_delta_rows"]:
            lines.append(
                f"| {name.replace('_', ' ')} | {row['rank']} | {row['step']} | {row['phase']} | "
                f"{row['matched_wait_layers']} | {row['delta_union_ms']:+.3f} |"
            )

    lines += ["", "| Comparison | Observed all-gather overlap | Matched GEMMs | Baseline median (µs) | "
              "Candidate median (µs) | Median paired Δ (µs) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, comparison in result["comparisons"].items():
        groups = {"all matched": comparison["all_matched_gemms"], **comparison["by_observed_overlap"]}
        for group, row in groups.items():
            values = (
                " | ".join(f"{row[key] / 1000:.3f}" for key in (
                    "baseline_median_ns", "candidate_median_ns", "paired_delta_median_ns",
                )) if row["matched"] else "— | — | —"
            )
            lines.append(f"| {name.replace('_', ' ')} | {group.replace('_', ' ')} | {row['matched']} | {values} |")
    lines += ["", "| Comparison | Baseline only | Candidate only | Kernel or geometry mismatch |",
              "| --- | ---: | ---: | ---: |"]
    for name, row in result["comparisons"].items():
        lines.append(f"| {name.replace('_', ' ')} | {row['unmatched_baseline']} | "
                     f"{row['unmatched_candidate']} | {row['kernel_mismatches']} |")
    followup = result.get("wait_diagnostic", {}).get("prefetch")
    if followup:
        lines += ["", "## Zero-CTA forward prefetch follow-up", ""]
        if followup["status"] == "verified":
            lines += [
                "Matched forward blocks exclude the root group and first block. Both captures use symmetric memory "
                "and zero-CTA.", "",
                f"Matched CE blocks {len(followup['matched_blocks'])}, "
                f"blocks with unresolved or non-CE route {len(followup['not_confirmed_ce_blocks'])}, "
                f"default-only blocks {followup['unmatched_default_blocks']}, "
                f"next-only blocks {followup['unmatched_next_blocks']}.", "",
                "| Metric | Paired blocks | Default median (ms) | Explicit next median (ms) | Paired median Δ (ms) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for field, row in followup["block_metric_medians"].items():
                delta = row["paired_delta_median_ms"]
                lines.append(
                    f"| {PREFETCH_METRICS[field]} | {row['matched_blocks']} | "
                    f"{ms(row['default_median_ms'])} | {ms(row['next_median_ms'])} | "
                    f"{f'{delta:+.3f}' if delta is not None else 'unmeasured'} |"
                )
            lines += ["", *(f"- {note}" for note in followup["notes"])]
        else:
            lines.append(f"Incomplete trace evidence. {followup['error']}")
    return "\n".join([*lines, "", *(f"- {note}" for note in result["notes"]), ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.run_dir)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n" if args.format == "json" else as_markdown(result)
        if args.output is None:
            print(content, end="")
        else:
            args.output.write_text(content)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"Invalid trace evidence: {error}\n")


if __name__ == "__main__":
    main()
