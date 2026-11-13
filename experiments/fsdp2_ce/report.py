"""Build an evidence-linked FSDP2 experiment report from downloaded Modal runs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import statistics
from urllib.parse import quote

from analyze_micro_trace import CASE_OPS, analyze as analyze_micro
from analyze_trace import analyze as analyze_training, compare
from check_smoke import NSYS_VERSION, audit, nonempty, successful, topology
from summarize import VARIANTS, as_markdown as timing_markdown, is_finite, read_json, require, summarize
from summarize_traces import PREFETCH_METRICS, summarize as summarize_traces

EVIDENCE_ERRORS = (KeyError, IndexError, TypeError, ValueError, OSError, sqlite3.Error)
DIAGNOSTICS = ("runtime.json", "nsys-version.txt", "nvidia-smi.txt", "topology.txt", "p2p-nvlink.txt",
               "nvlink-status.txt", "pip-freeze.txt", "source-hashes.json")
COMMON_WORKERS = {
    "experiments/fsdp2_ce/bench.py",
    "experiments/fsdp2_ce/analyze_trace.py",
}


def link(path, base, label=None):
    return f"[{label or path.name}]({quote(os.path.relpath(path, base))})"


def links(root, names, base):
    return " · ".join(link(root / name, base) for name in names if (root / name).is_file())


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def route_counts(routes):
    return ", ".join(f"{route} {count}" for route, count in sorted(routes.items())) or "none observed"


def stack_lines(label, root, runtime, nvlink, base):
    paths = [item["artifact"] for item in nvlink["nvlink_evidence"]]
    evidence = links(root, dict.fromkeys(paths), base)
    nvlink_scope = "Pairwise NVLink confirmed." if nvlink["nvlink_domain_confirmed"] else "Pairwise NVLink unverified."
    nsys = (root / "nsys-version.txt").read_text().strip()
    return [
        f"- **{label}.** {' / '.join(runtime['gpu_names'])}, driver {nvlink['driver_version']}, "
        f"PyTorch {runtime['torch']}, CUDA {runtime['cuda']}, NCCL {'.'.join(map(str, runtime['nccl']))}, "
        f"`{nsys}`. {nvlink_scope} {evidence}".rstrip(),
        f"  GPU UUIDs `{' / '.join(nvlink['gpu_uuids'])}`. {links(root, DIAGNOSTICS, base)}",
    ]


def source_links(root, base):
    hashes = read_json(root / "source-hashes.json")
    require(bool(hashes), f"{root.name}: source manifest is empty")
    sources = []
    for name, digest in hashes.items():
        path = nonempty(root / "source" / name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"{root.name}: source hash mismatch for {name}")
        if name.startswith("experiments/fsdp2_ce/"):
            sources.append(link(path, base))
    return hashes, sources


def run_gpu_model(runtime):
    names = runtime["gpu_names"]
    require(len(names) == 2 and names[0] == names[1], "Expected two GPUs of the same model")
    match = re.search(r"\b(B200|B300)\b", names[0])
    require(match is not None, f"Unexpected GPU model {names[0]!r}")
    require(runtime.get("gpu_model", match[1]) == match[1], "Requested GPU model differs from runtime")
    return match[1]


def workload_hashes(hashes):
    # The launcher can add a dispatch option between runs without changing
    # the benchmark, model implementation or event interpretation.
    return {
        name: digest for name, digest in hashes.items()
        if name in COMMON_WORKERS or not name.startswith("experiments/fsdp2_ce/")
    }


def smoke_lines(smoke, root, base):
    lines = [
        "## Smoke verification", "",
        f"The {smoke['config']['parameters']:,}-parameter model passed the dual-rank numerical checks. "
        "The analyzer re-read each SQLite and associated NCCL CE AllGather with CUDA GPU copies on both zero-CTA ranks.",
        "", "| Variant | Rank 0 routes | Rank 1 routes | Raw SQLite |",
        "| --- | --- | --- | --- |",
    ]
    for variant in VARIANTS:
        row = smoke["traces"][variant]
        routes = [route_counts(row["ranks"][rank]["gather_routes"]) for rank in ("0", "1")]
        artifact = link(root / f"trace-{variant}" / "profile.sqlite", base)
        lines.append(f"| {variant} | {routes[0]} | {routes[1]} | {artifact} |")
    recovery = smoke.get("recovered_analysis")
    if recovery:
        lines += [
            "", f"The original `{recovery['case']}` analysis exited with code "
            f"{recovery['original_analysis_returncode']}. The corrected CE event matcher re-analyzed the saved SQLite. "
            "The original worker, export, launcher status and failed analysis are preserved.",
            links(root, ("summary.json", "trace-zero/status.json", "trace-zero/analysis.log"), base),
        ]
    return lines


def parity_lines(timing):
    lines = [
        "## XL numerical check", "",
        "| Variant | Rank losses | Max abs loss error | Max abs gradient error | Max abs parameter error |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        row = timing["parity"][variant]
        losses = " / ".join(fmt(value, 6) for value in row["rank_losses"])
        errors = [
            row["max_abs_loss_error"], row["gradients"]["max_abs_error"], row["parameters"]["max_abs_error"],
        ]
        fields = " | ".join(f"{value:.3g}" if value is not None else "reference" for value in errors)
        lines.append(f"| {variant} | {losses} | {fields} |")
    tolerances = timing["parity"]["default"]["tolerances"]
    criteria = ", ".join(
        f"{field} (rtol {value['rtol']:g}, atol {value['atol']:g})" for field, value in tolerances.items()
    )
    lines += ["", f"Compared {timing['config']['parameters']:,} elements each for reconstructed gradients and "
              f"post-AdamW parameters. Tolerances were {criteria}."]
    return lines


def training_trace_lines(trace, raw_traces):
    lines = [
        "## XL GPU traces", "",
        "Each variant has two captured steps per rank. The table shows exposed all-gather wait on the slowest "
        "profiled rank per step and route counts across both ranks.",
        "", "| Variant | Step | Rank | Forward wait union (ms) | Backward wait union (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, report in trace["variants"].items():
        for row in report["slowest_profiled"]:
            step = row["step"]
            for rank in row["ranks"]:
                phases = report["ranks"][str(rank)]["steps"][step]["phases"]
                lines.append(f"| {variant} | {step} | {rank} | {fmt(phases['forward']['wait_union_ms'])} | "
                             f"{fmt(phases['backward']['wait_union_ms'])} |")
    lines += ["", "| Variant | Rank 0 routes | Rank 1 routes |", "| --- | --- | --- |"]
    for variant, report in trace["variants"].items():
        routes = [route_counts(report["ranks"][rank]["gather_routes"]) for rank in ("0", "1")]
        lines.append(f"| {variant} | {routes[0]} | {routes[1]} |")
    if all(set(row["gather_routes"]) == {"copy_engine"} for row in trace["variants"]["symm"]["ranks"].values()):
        lines += [
            "", "NCCL 2.29.7 can select CE with DEFAULT CTA policy for eligible symmetric-memory all-gathers "
            "on Blackwell with direct NVLink. The code compares the NCCL element count with `NCCL_SYM_CE_THRESHOLD`, "
            "whose default is 8,388,608. The XL root and block payloads exceeded this threshold. "
            "[Automatic selection in NCCL 2.29.7](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L2775-L2779) · "
            "[Threshold definition](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L28)",
        ]
    lines += [
        "", "Each row below accounts for the waits outside the next-layer-prefetch trigger. "
        "That trigger covers forward blocks 01–31 and all 32 backward block gathers on the slowest profiled rank.",
        "", "| Variant | Rank | Step | Root wait (ms) | First block wait (ms) | Blocks 01–31 wait sum (ms) | "
        "Backward blocks wait sum (ms) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, report in raw_traces.items():
        for rank, data in report["ranks"].items():
            for step in data["steps"]:
                waits = {
                    (layer["phase"], layer["layer"]): layer["measured_wait"]["exposed_wait_ms"]
                    for layer in step["layers"] if layer["measured_wait"] is not None
                }
                root = waits["forward", "root"]
                first = waits["forward", "00"]
                later = sum(waits["forward", f"{block:02}"] for block in range(1, 32))
                backward = sum(waits["backward", f"{block:02}"] for block in range(32))
                lines.append(f"| {variant} | {rank} | {step['step']} | {fmt(root)} | {fmt(first)} | "
                             f"{fmt(later)} | {fmt(backward)} |")
    lines += [
        "", "| Comparison | Matched GEMMs | Overlap in both | Overlap only before | Overlap only after | "
        "Median paired Δ with overlap in both (µs) |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in trace["comparisons"].items():
        groups = row["by_observed_overlap"]
        delta = groups["both"]["paired_delta_median_ns"]
        lines.append(f"| {name.replace('_', ' ')} | {row['all_matched_gemms']['matched']} | "
                     f"{groups['both']['matched']} | {groups['baseline_only']['matched']} | "
                     f"{groups['candidate_only']['matched']} | {fmt(delta / 1000 if delta is not None else None)} |")
    lines += [
        "", "| CE policy comparison | Rank | Matched GEMMs | Overlap in both | All-pair median Δ (µs) | "
        "Both-overlap median Δ (µs) |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    pairs = compare(raw_traces["zero"], raw_traces["symm"])["pairs"]
    for rank in ("0", "1"):
        rows = [row for row in pairs if row["rank"] == rank]
        both = [row for row in rows if row["baseline_overlap_ns"] > 0 and row["candidate_overlap_ns"] > 0]
        medians = [statistics.median(row["delta_ns"] for row in group) / 1000 if group else None
                   for group in (rows, both)]
        lines.append(f"| zero vs symm | {rank} | {len(rows)} | {len(both)} | {fmt(medians[0])} | {fmt(medians[1])} |")
    lines += ["", *trace["notes"]]
    return lines


def median_slowest_waits(report, phase):
    totals = []
    for row in report["slowest_profiled"]:
        phase_totals = [
            report["ranks"][str(rank)]["steps"][row["step"]]["phases"][phase]["wait_union_ms"]
            for rank in row["ranks"]
        ]
        require(all(value is not None for value in phase_totals), "Missing slowest-rank wait sample")
        totals.append(max(phase_totals))
    return statistics.median(totals)


def conclusion(timing, trace):
    default_ms = timing["variants"]["default"]["timing"]["median_ms"]
    symm_ms = timing["variants"]["symm"]["timing"]["median_ms"]
    zero_ms = timing["variants"]["zero"]["timing"]["median_ms"]
    ratios = timing["rotation_zero_to_symm_median_ratios"]
    symm, zero = (trace["variants"][variant] for variant in ("symm", "zero"))
    waits = [
        f"{phase} {median_slowest_waits(symm, phase):.3f} → {median_slowest_waits(zero, phase):.3f} ms"
        for phase in ("forward", "backward")
    ]
    compared = trace["comparisons"]["zero_vs_symm"]["by_observed_overlap"]["both"]
    same_routes = all(
        set(row["gather_routes"]) == {"copy_engine"}
        for variant in ("symm", "zero") for row in trace["variants"][variant]["ranks"].values()
    )
    route = (
        "Both symmetric-memory variants used copy-engine all-gather on both ranks. "
        "This comparison measures the incremental CTA policy change."
        if same_routes else "The per-rank trace tables give the observed communication routes."
    )
    gemm = (
        f"The median paired GEMM duration change was {compared['paired_delta_median_ns'] / 1000:+.3f} µs "
        f"across {compared['matched']} pairs with overlap in both captures."
        if compared["matched"] else "No matched GEMM pair overlapped an observed all-gather in both captures."
    )
    primary = trace["comparisons"]["symm_vs_default"]["by_observed_overlap"]["both"]
    primary_gemm = (
        f"Across {primary['matched']} GEMM pairs with overlap in both captures, the paired median "
        f"duration change was {primary['paired_delta_median_ns'] / 1000:+.3f} µs."
        if primary["matched"] else "No matched GEMM pair overlapped an observed all-gather in both captures."
    )
    return [
        f"Enabling symmetric memory with default CTA policy changed the unprofiled median step from "
        f"{default_ms:.3f} ms to {symm_ms:.3f} ms ({symm_ms - default_ms:+.3f} ms). The all-gather route "
        f"also changed in the GPU trace. {primary_gemm} This comparison includes the staging-buffer change.", "",
        f"Relative to symmetric memory with default CTA policy, zero-CTA changed the unprofiled median training "
        f"step from {symm_ms:.3f} ms to {zero_ms:.3f} ms ({zero_ms - symm_ms:+.3f} ms). "
        f"The three paired rotation ratios were {', '.join(f'{ratio:.4f}×' for ratio in ratios)}. "
        f"{route}", "",
        f"Median exposed wait on the slowest profiled rank was {waits[0]} and {waits[1]}. "
        f"{gemm} These waits describe the captured steps.", "",
    ]


def read_micro_samples(root, run, variant, mode, runtime):
    label = f"micro-{mode}-{variant}"
    folder = root / label
    case = read_json(folder / "status.json")
    require(case in run["cases"] and case["label"] == label and case["variant"] == variant
            and case["mode"] == mode and case["workload"] == "micro" and case["status"] == "ok",
            f"{label}: case metadata disagrees")
    successful(case["worker"], label)
    steps = 15 if mode == "bench" else 3
    rows = [read_json(folder / f"rank{rank}.json") for rank in range(2)]
    d_model, d_ff = 2560, 10240
    block_bytes = 4 * (4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model)
    for rank, row in enumerate(rows):
        expected = {
            "rank": rank, "gpu": runtime["gpu_names"][rank], "variant": variant, "warmup": 5,
            **{key: runtime[key] for key in ("torch", "cuda", "nccl")},
            "profiled": mode == "trace", "nccl_cta_policy": "ZERO" if variant == "zero" else "DEFAULT",
            "projection": "q", "input_shape": [2, 512, d_model], "weight_shape": [d_model, d_model],
            "dtype": "torch.float32", "tf32": False, "gather_result_bytes": block_bytes,
            "gather_local_shard_bytes": block_bytes // 2,
        }
        require(all(row[key] == value for key, value in expected.items()), f"{label}: rank {rank} protocol mismatch")
        for name, operations in CASE_OPS.items():
            samples = [item for item in row["samples"] if item["case"] == name]
            require(sorted(item["step"] for item in samples) == list(range(steps)), f"{label}: rank {rank} {name} coverage")
            for sample in samples:
                require(all(is_finite(sample[f"{op}_ms"]) and sample[f"{op}_ms"] > 0 for op in operations),
                        f"{label}: rank {rank} {name} invalid duration")
        require(len(row["samples"]) == steps * len(CASE_OPS), f"{label}: unexpected rank {rank} samples")
    if mode == "bench":
        return rows
    for stage in ("export", "analysis"):
        successful(case[stage], f"{label}/{stage}")
    nonempty(folder / "profile.nsys-rep")
    trace = analyze_micro(nonempty(folder / "profile.sqlite"))
    require(trace["two_rank_case_coverage"], f"{label}: insufficient GPU evidence")
    if variant == "zero":
        require(trace["copy_engine_confirmed_all_gathers"], f"{label}: some gathers lack confirmed CE")
    for data in trace["ranks"].values():
        require(all(len(group["samples"]) == steps for group in data["cases"].values()),
                f"{label}: unexpected GPU sample coverage")
        if variant == "symm":
            require(all(set(group["route_counts"]) == {"sm_kernel"} for name, group in data["cases"].items()
                        if "gather" in CASE_OPS[name]), f"{label}: symmetric control route unresolved")
    return trace


def micro_lines(root, run, runtime):
    lines = [
        "## Same-size all-gather / GEMM follow-up", "",
        "The control packs a staff XL block before timing and runs one FP32 Q projection on a separate CUDA stream. "
        "CUDA event medians are from 15 unprofiled samples per case and rank. GPU intersections come from three separate trace samples.",
        "", "| Variant | Rank | Case | Gather median (ms) | GEMM median (ms) | GPU intersection median (µs) | Positive overlaps |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("symm", "zero"):
        rows = read_micro_samples(root, run, variant, "bench", runtime)
        trace = read_micro_samples(root, run, variant, "trace", runtime)
        for rank, row in enumerate(rows):
            for name, operations in CASE_OPS.items():
                samples = [item for item in row["samples"] if item["case"] == name]
                values = {
                    op: statistics.median(item[f"{op}_ms"] for item in samples) if op in operations else None
                    for op in ("gather", "gemm")
                }
                case = trace["ranks"][str(rank)]["cases"][name]
                overlap = case["gpu_intersection_ns"]["median"]
                positive = f"{case['positive_intersection_samples']}/{len(case['samples'])}" if overlap is not None else "—"
                lines.append(f"| {variant} | {rank} | {name} | {fmt(values['gather'])} | {fmt(values['gemm'])} | "
                             f"{fmt(overlap / 1000 if overlap is not None else None)} | {positive} |")
    lines += ["", "This control tests whether matched operations slow down when they overlap. "
              "Link saturation or HBM contention requires resource-specific evidence."]
    return lines


def followup_lines(root, run, timing, trace, base, issues):
    decision_path = root / "wait-diagnostic.json"
    decision = read_json(decision_path)
    selected = decision["followup"]
    trigger = decision["median_wait_ms"]
    threshold = decision["threshold_ms"]
    require(is_finite(threshold) and threshold > 0, "Invalid diagnostic threshold")
    expected = None
    if trigger["forward"] is not None and trigger["forward"] >= threshold:
        expected = "prefetch"
    elif trigger["backward"] is not None and trigger["backward"] >= threshold:
        expected = "micro"
    require(selected == expected, "Recorded follow-up disagrees with its trigger measurements")
    lines = ["## Follow-up decision", "",
             f"The zero-CTA slowest-rank median exposed wait was {fmt(trigger['forward'])} ms across prefetchable forward "
             f"blocks and {fmt(trigger['backward'])} ms in backward. The trigger was {decision['threshold_ms']:g} ms. "
             f"Selected control was {selected or 'none'}. {link(decision_path, base)}"]
    if selected == "prefetch":
        plain = timing.get("wait_diagnostic", {}).get("prefetch", {})
        profiled = trace.get("wait_diagnostic", {}).get("prefetch", {})
        require(plain.get("status") == profiled.get("status") == "verified",
                "Forward prefetch plain timing, parity, or paired trace evidence is incomplete")
        lines += [
            "", f"Explicit next-layer prefetch passed parity and changed the unprofiled median by "
            f"{plain['median_delta_ms']:+.3f} ms ({plain['median_ratio']:.4f}×).",
            "", "| Paired block metric | Blocks | Default median (ms) | Next median (ms) | Median paired Δ (ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for field, row in profiled["block_metric_medians"].items():
            delta = row["paired_delta_median_ms"]
            lines.append(f"| {PREFETCH_METRICS[field]} | {row['matched_blocks']} | "
                         f"{fmt(row['default_median_ms'])} | {fmt(row['next_median_ms'])} | "
                         f"{f'{delta:+.3f}' if delta is not None else '—'} |")
        lines += ["", *profiled["notes"]]
    elif selected == "micro":
        lines += ["", *micro_lines(root, run, timing["runtime"])]
    else:
        require(selected is None, f"Unknown diagnostic control {selected!r}")
        if any(trigger[phase] is None for phase in ("forward", "backward")):
            issues.append("Follow-up threshold cannot be evaluated from the measured wait markers")
    return lines


def verify_xl_trace_coverage(trace):
    for rank in ("0", "1"):
        devices = {tuple(row["ranks"][rank]["devices"]) for row in trace["variants"].values()}
        require(len(devices) == 1, f"XL trace rank {rank} device differs across variants")
    for variant, data in trace["variants"].items():
        for rank, row in data["ranks"].items():
            routes = row["gather_routes"]
            require(routes and row["unmatched_wait_records"] == 0, f"{variant} rank {rank}: missing gather or wait linkage")
            require("unresolved" not in routes and set(routes) <= {"sm_kernel", "copy_engine", "mixed"},
                    f"{variant} rank {rank}: unresolved all-gather route")
            if variant == "default":
                require(set(routes) == {"sm_kernel"}, f"{variant} rank {rank}: expected SM gathers")
            if variant == "zero":
                require(set(routes) == {"copy_engine"}, f"{variant} rank {rank}: expected CE gathers")
            for step in row["steps"]:
                require(all(phase["wait_markers"] > 0 for phase in step["phases"].values()),
                        f"{variant} rank {rank} step {step['step']}: missing forward or backward wait markers")
    pairs = trace["comparisons"]["zero_vs_symm"]
    require(pairs["all_matched_gemms"]["matched"] > 0 and pairs["wait_delta_rows"],
            "No matched GEMM or wait coverage for zero-CTA versus symmetric memory")


def verify_trace_source(root, label):
    folder = root / label
    recorded = read_json(folder / "trace.json")
    observed = analyze_training(nonempty(folder / "profile.sqlite"), folder, expected_ranks=2)
    keys = ("ranks", "two_rank_gpu_coverage", "copy_engine_confirmed_all_ranks",
            "copy_engine_only_all_gathers", "slowest_profiled_rank_by_step")
    # Saved JSON object keys are strings. Normalize step keys from the live analysis.
    normalized = json.loads(json.dumps(observed))
    for key in keys:
        require(recorded[key] == normalized[key], f"{label}: {key} differs from raw SQLite reanalysis")
    log = nonempty(folder / "worker.log").read_text()
    policy = "ZERO to 2" if label.startswith("trace-zero") else "DEFAULT to 0"
    require(f"Parsed environment variable NCCL_CTA_POLICY={policy}" in log,
            f"{label}: NCCL worker log does not corroborate CTA policy")
    ce_routes = any({"copy_engine", "mixed"} & set(row["gather_routes"]) for row in observed["ranks"].values())
    require(("AllGather [Copy Engine]" in log) == ce_routes,
            f"{label}: NCCL worker log and observed all-gather route disagree")
    return observed


def teardown_note(root, base):
    artifacts = []
    for variant in ("symm", "zero"):
        label = f"trace-{variant}"
        folder = root / label
        path = folder / "worker.log"
        log = path.read_text()
        first_warning = log.find("corrupted comm object detected")
        if first_warning < 0:
            continue
        require(log[:first_warning].count("Destroy COMPLETE") >= 2 and "ncclCommWindowDeregister failed" in log[first_warning:],
                f"{label}: ambiguous communicator teardown warning")
        successful(read_json(folder / "status.json")["worker"], label)
        artifacts.append(link(path, base, f"{label} log"))
    if artifacts:
        return [
            "", "The symmetric-memory trace workers reported window deregistration warnings after both communicators "
            "logged `Destroy COMPLETE`. Both workers exited 0 and the measurement and export checks passed. "
            + " · ".join(artifacts),
        ]
    return []


def command_index(label, root, run, base):
    summary = root / "summary.json"
    if not run:
        return [f"- {label} manifest unavailable at `{root}`."]
    names = ("status.json", "worker.log", "rank0.json", "rank1.json", "profile.nsys-rep", "profile.sqlite",
             "trace.json", "export.log", "analysis.log", "nccl-topology.xml")
    rows = [f"### {label}", "", link(summary, base), ""]
    for case in run["cases"]:
        folder = root / case["label"]
        require(case == read_json(folder / "status.json"), f"{case['label']}: recorded status differs from case manifest")
        rows += [f"#### {case['label']}", "", links(folder, names, base), "", "```sh"]
        for stage in ("worker", "export", "analysis"):
            if stage not in case:
                continue
            call = case[stage]
            env = [f"{name}={value}" for name, value in call["environment"].items()]
            rows.append(shlex.join((["env", *env] if env else []) + call["command"]))
        rows += ["```", ""]
    return rows


def build(smoke_root, full_root, base, *, recover_smoke_analysis=False):
    sections = []
    issues = []
    runs = {}
    provenance = {}
    smoke = None
    try:
        if recover_smoke_analysis:
            smoke = audit(smoke_root, recover_analysis=True)
        else:
            smoke = audit(smoke_root)
    except EVIDENCE_ERRORS as error:
        issues.append(f"Smoke audit: {error}")
    for label, root in (("Smoke", smoke_root), ("XL", full_root)):
        try:
            runs[label] = read_json(root / "summary.json")
            recovered = label == "Smoke" and smoke is not None and smoke.get("recovered_analysis")
            require(runs[label]["status"] == "ok" or recovered,
                    f"launcher reported {runs[label]['status']}: {runs[label].get('error')}")
            runtime = read_json(root / "runtime.json")
            gpu = run_gpu_model(runtime)
            nsys = nonempty(root / "nsys-version.txt").read_text().strip()
            require(re.search(rf"\bversion {re.escape(NSYS_VERSION)}(?:-|\s|$)", nsys), "Unexpected Nsight version")
            nvlink = topology(root, runtime, gpu)
            if not nvlink["nvlink_domain_confirmed"]:
                issues.append(f"{label}: pairwise NVLink could not be verified")
            sections.extend(stack_lines(label, root, runtime, nvlink, base))
            hashes, files = source_links(root, base)
            sections.append("  Verified source snapshots. " + " · ".join(files))
            provenance[label] = {
                **{key: runtime[key] for key in ("torch", "cuda", "nccl", "gpu_names")},
                "gpu": gpu, "nsys": nsys, "cuda_trace": runs[label]["cuda_trace"],
                "driver_version": nvlink["driver_version"], "hashes": hashes, "recovered_analysis": bool(recovered),
            }
        except EVIDENCE_ERRORS as error:
            issues.append(f"{label} runtime or source evidence: {error}")
    if len(provenance) == 2:
        for key in ("torch", "cuda", "nccl", "nsys", "cuda_trace", "driver_version", "gpu_names"):
            if provenance["Smoke"][key] != provenance["XL"][key]:
                issues.append(f"Smoke and XL runs differ in {key}")
        smoke_hashes = workload_hashes(provenance["Smoke"]["hashes"])
        xl_hashes = workload_hashes(provenance["XL"]["hashes"])
        for name in sorted(smoke_hashes.keys() | xl_hashes.keys()):
            if provenance["Smoke"]["recovered_analysis"] and name == "experiments/fsdp2_ce/analyze_trace.py":
                continue
            if smoke_hashes.get(name) != xl_hashes.get(name):
                issues.append(f"Smoke and XL source snapshots differ for {name}")
    sections = ["## Hardware and provenance", "", *sections]

    if smoke is not None:
        sections += ["", *smoke_lines(smoke, smoke_root, base)]

    timing = trace = None
    try:
        timing = summarize(full_root)
        config = timing["config"]
        sections += [
            "", *parity_lines(timing), "", "## Unprofiled XL training steps", "",
            f"Model {config['parameters']:,} parameters, FP32, global batch {config['global_batch']}, sequence "
            f"{config['context_length']}, AdamW. Block sharding and default prefetch were fixed across configurations.",
            "", *timing_markdown(timing).splitlines()[4:],
        ]
    except EVIDENCE_ERRORS as error:
        issues.append(f"XL numerical or timing evidence: {error}")

    try:
        trace = summarize_traces(full_root)
        verify_xl_trace_coverage(trace)
        raw_traces = {variant: verify_trace_source(full_root, f"trace-{variant}") for variant in VARIANTS}
        followup = trace.get("wait_diagnostic", {}).get("prefetch")
        if followup and followup["status"] == "verified":
            for label in followup["cases"].values():
                verify_trace_source(full_root, label)
        sections += ["", *training_trace_lines(trace, raw_traces), *teardown_note(full_root, base)]
    except EVIDENCE_ERRORS as error:
        issues.append(f"XL GPU trace evidence: {error}")

    if timing is not None and trace is not None and "XL" in runs:
        try:
            sections += ["", *followup_lines(full_root, runs["XL"], timing, trace, base, issues)]
        except EVIDENCE_ERRORS as error:
            issues.append(f"XL follow-up evidence: {error}")

    sections += ["", "## Reproduction and raw artifacts", "",
                 "The launcher recorded the arguments below. Their paths refer to the Modal container. "
                 "Each case links per-rank JSON and logs, with the original Nsight report and SQLite export for traces.", ""]
    if provenance:
        sections += ["Run from the Assignment 2 directory with the configured Modal account.", "", "```sh"]
        for label, stage in (("Smoke", "smoke"), ("XL", "full")):
            if label in provenance:
                spec = provenance[label]
                sections.append(
                    f"FSDP2_CE_GPU_MODEL={spec['gpu']} modal run --detach "
                    f"experiments/fsdp2_ce/modal_run.py --stage {stage} --cuda-trace {spec['cuda_trace']}"
                )
        sections += ["```", ""]
    for label, root in (("Smoke", smoke_root), ("XL", full_root)):
        try:
            sections.extend(command_index(label, root, runs.get(label), base))
        except EVIDENCE_ERRORS as error:
            sections.append(f"Command index incomplete for {label}. {error}")
            issues.append(f"{label} command provenance: {error}")

    header = ["# CS336 A2 · FSDP2 copy-engine all-gather", "",
              "**Evidence status: incomplete.**" if issues else "**Evidence status: verified.**", ""]
    if issues:
        header += [*(f"- {issue}" for issue in issues), ""]
    else:
        header += conclusion(timing, trace)
    return "\n".join([*header, *sections, ""]), issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True, help="Downloaded smoke run folder")
    parser.add_argument("--full", type=Path, required=True, help="Downloaded XL run folder")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report path")
    parser.add_argument("--recover-smoke-analysis", action="store_true",
                        help="Allow offline smoke verification after the final launcher analysis failed")
    args = parser.parse_args()
    report, issues = build(args.smoke.resolve(), args.full.resolve(), args.output.resolve().parent,
                           recover_smoke_analysis=args.recover_smoke_analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    if issues:
        parser.exit(2, f"Incomplete report written to {args.output}: {issues[0]}\n")


if __name__ == "__main__":
    main()
