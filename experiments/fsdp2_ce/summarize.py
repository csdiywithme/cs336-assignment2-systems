"""Validate and aggregate plain timing and parity from a downloaded full FSDP2 run."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics


VARIANTS = ("default", "symm", "zero")
ROTATIONS = tuple(VARIANTS[index:] + VARIANTS[:index] for index in range(len(VARIANTS)))
STEPS = 12
WARMUP = 5
GIB = 2**30
GPU_STACKS = {
    "B200": {"torch": "2.12.1+cu130", "cuda": "13.0", "nccl": [2, 29, 7]},
    "B300": {"torch": "2.12.1+cu132", "cuda": "13.2", "nccl": [2, 29, 7]},
}
TOLERANCES = {
    "loss": {"rtol": 2e-6, "atol": 2e-6},
    "gradient": {"rtol": 2e-5, "atol": 1e-7},
    "parameter": {"rtol": 2e-5, "atol": 2e-6},
}
XL_CONFIG = {
    "parameters": 3_406_809_600, "d_model": 2560, "d_ff": 10240,
    "num_layers": 32, "num_heads": 32, "vocab_size": 10000, "context_length": 512,
    "global_batch": 4, "dtype": "float32", "activation_checkpointing": False,
    "tf32": False, "optimizer": "torch.optim.AdamW", "adamw_lr": 1e-3,
    "adamw_foreach": False, "adamw_fused": False,
    "reshard_after_forward": {"block": True, "root": False},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def validate_runtime(runtime):
    names = runtime["gpu_names"]
    require(isinstance(names, list) and len(names) == 2, "Expected two GPU ranks")
    models = [re.search(r"\bB(?:200|300)\b", name) for name in names]
    require(all(models) and models[0][0] == models[1][0], "Expected two B200 or two B300 GPUs")
    model = models[0][0]
    require(runtime.get("gpu_model", model) == model, "Declared GPU model disagrees with both rank devices")
    stack = GPU_STACKS[model]
    require(all(runtime.get(key) == value for key, value in stack.items()), f"Unexpected PyTorch/CUDA/NCCL stack for {model}")
    require(runtime["cuda_peer_access"] == [True, True] and runtime["symm_api"] is True,
            "Expected bidirectional CUDA P2P and the FSDP2 symmetric-memory API")
    return model, stack


def is_finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def distribution(values):
    return {
        "count": len(values), "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values), "stdev_ms": statistics.stdev(values),
        "min_ms": min(values), "p95_ms": statistics.quantiles(values, n=20, method="inclusive")[-1],
        "max_ms": max(values),
    }


def load_case(root, label, variant, mode, runtime, config, *, prefetch="default"):
    folder = root / label
    status = read_json(folder / "status.json")
    expected_status = {"label": label, "variant": variant, "mode": mode, "workload": "xl", "status": "ok"}
    require(all(status[key] == value for key, value in expected_status.items()), f"{label}: invalid case status")
    require(status["worker"]["returncode"] == 0 and status["worker"]["timed_out"] is False,
            f"{label}: worker did not finish")
    result = [read_json(folder / f"rank{rank}.json") for rank in range(2)]
    expected = {
        "variant": variant, "mode": mode, "workload": "xl", "world_size": 2,
        "seed": 2026, "forward_prefetch": prefetch,
        "nccl_cta_policy": "ZERO" if variant == "zero" else "DEFAULT",
        "torch": runtime["torch"], "cuda": runtime["cuda"], "nccl": runtime["nccl"],
        "config": config,
    }
    for rank, row in enumerate(result):
        rank_expected = expected | {"rank": rank, "gpu": runtime["gpu_names"][rank]}
        for key, value in rank_expected.items():
            require(row[key] == value, f"{label}/rank{rank}.json: unexpected {key}")
    return result


def tensor_check(rows, recording, parameters, label):
    shape_by_name = {}
    max_error = None if recording else 0
    for row in rows:
        name, shape = row["name"], row["shape"]
        require(name not in shape_by_name and isinstance(shape, list) and
                all(type(size) is int and size > 0 for size in shape) and
                row["elements"] == math.prod(shape), f"{label}: invalid tensor {name}")
        shape_by_name[name] = shape
        require(row["finite"] is True and "error" not in row, f"{label}: invalid value for {name}")
        if not recording:
            error = row["max_abs_error"]
            require(row["mismatched_elements"] == 0 and is_finite(error) and error >= 0,
                    f"{label}: parity failure for {name}")
            max_error = max(max_error, error)
    elements = sum(math.prod(shape) for shape in shape_by_name.values())
    require(elements == parameters, f"{label}: covered {elements} of {parameters} model elements")
    return shape_by_name, {"tensors": len(rows), "elements": elements, "max_abs_error": max_error}


def check_case(root, label, variant, runtime, config, *, prefetch="default", reference_shapes=None):
    rows = load_case(root, label, variant, "check", runtime, config, prefetch=prefetch)
    check = rows[0]["check"]
    recording = variant == "default" and prefetch == "default"
    require(all(row["passed"] is True for row in rows) and check["passed"] is True
            and check["recorded_reference"] is recording, f"{label}: failed rank parity or reference status")
    require(len(check["rank_losses"]) == 2 and all(is_finite(value) for value in check["rank_losses"]),
            f"{label}: missing or nonfinite rank loss")
    tolerances = check["tolerances"]
    require(tolerances == TOLERANCES, f"{label}: unexpected parity tolerances")
    loss = check["loss_check"]
    require(loss["finite"] is True, f"{label}: nonfinite loss")
    if not recording:
        require(loss["mismatched_elements"] == 0 and is_finite(loss["max_abs_error"]) and
                loss["max_abs_error"] >= 0, f"{label}: loss parity failure")
    grad_shapes, grads = tensor_check(check["gradients"], recording, config["parameters"], f"{label}/gradient")
    param_shapes, params = tensor_check(check["parameters"], recording, config["parameters"], f"{label}/parameter")
    require(grad_shapes == param_shapes, f"{label}: gradient and parameter inventories differ")
    require(recording or param_shapes == reference_shapes, f"{label}: candidate does not match reference inventory")
    return {
        "case": label, "passed": True, "recorded_reference": recording,
        "rank_losses": check["rank_losses"], "max_abs_loss_error": loss.get("max_abs_error"),
        "gradients": grads, "parameters": params, "tolerances": tolerances,
    }, param_shapes


def bench_case(root, label, variant, runtime, config, *, prefetch="default"):
    rows = load_case(root, label, variant, "bench", runtime, config, prefetch=prefetch)
    memory = []
    for rank, row in enumerate(rows):
        require(row["profiled"] is False and row["warmup"] == WARMUP and len(row["samples"]) == STEPS,
                f"{label}/rank{rank}.json: unexpected measurement protocol")
        for step, sample in enumerate(row["samples"]):
            require(sample["rank"] == rank and sample["step"] == step and
                    is_finite(sample["duration_ms"]) and sample["duration_ms"] > 0 and is_finite(sample["loss"]),
                    f"{label}/rank{rank}.json: invalid step {step}")
            allocated, reserved = sample["peak_allocated_bytes"], sample["peak_reserved_bytes"]
            require(type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved,
                    f"{label}/rank{rank}.json: invalid memory for step {step}")
        nvml = row["device_memory"]
        observed = nvml["observed_max_device_bytes"]
        require((observed is None and nvml["samples"] == 0) or
                (type(observed) is int and observed > 0 and nvml["samples"] > 0),
                f"{label}/rank{rank}.json: invalid NVML metadata")
        memory.append({
            "rank": rank, "peak_allocated_bytes": max(sample["peak_allocated_bytes"] for sample in row["samples"]),
            "peak_reserved_bytes": max(sample["peak_reserved_bytes"] for sample in row["samples"]),
            "nvml": nvml,
        })

    steps = []
    for rank0, rank1 in zip(*(row["samples"] for row in rows), strict=True):
        slowest = max((rank0, rank1), key=lambda sample: sample["duration_ms"])
        steps.append({
            "step": slowest["step"], "rank_ms": [rank0["duration_ms"], rank1["duration_ms"]],
            "slowest_rank": slowest["rank"], "max_rank_ms": slowest["duration_ms"],
        })
    return {"case": label, "steps": steps, "timing": distribution([step["max_rank_ms"] for step in steps]),
            "memory_by_rank": memory}


def allocator_peaks(memory):
    return {key: max(row[key] for row in memory)
            for key in ("peak_allocated_bytes", "peak_reserved_bytes")}


def wait_followup(root, run, runtime, config, reference_shapes):
    path = root / "wait-diagnostic.json"
    if not path.is_file():
        return {}
    evidence = {"path": path.name, "trace_source": "trace-zero/trace.json"}
    try:
        decision = read_json(path)
        evidence["decision"] = decision
        if decision["followup"] != "prefetch":
            return {"wait_diagnostic": evidence}

        result = {"status": "incomplete", "reference_case": "check-default"}
        evidence["prefetch"] = result
        required = ("check-zero-next", "bench-zero-default", "bench-zero-next")
        recorded = [case for case in run["cases"] if case["label"] in required]
        require(tuple(case["label"] for case in recorded) == required and
                all(case["status"] == "ok" for case in recorded),
                "Missing, failed, or misordered launcher prefetch cases")
        parity, _ = check_case(root, "check-zero-next", "zero", runtime, config,
                               prefetch="next", reference_shapes=reference_shapes)
        cases = {prefetch: bench_case(root, f"bench-zero-{prefetch}", "zero", runtime, config, prefetch=prefetch)
                 for prefetch in ("default", "next")}
        for case in cases.values():
            case.update(allocator_peaks(case["memory_by_rank"]))
        default_ms = cases["default"]["timing"]["median_ms"]
        next_ms = cases["next"]["timing"]["median_ms"]
        result.update(status="verified", parity=parity, cases=cases, median_delta_ms=next_ms - default_ms,
                      median_ratio=next_ms / default_ms)
    except (KeyError, OSError, TypeError, ValueError) as error:
        evidence["error"] = str(error)
    return {"wait_diagnostic": evidence}


def summarize(root):
    run = read_json(root / "summary.json")
    require(run["stage"] == "full", "Expected the full-stage output directory")
    runtime = read_json(root / "runtime.json")
    gpu_model, _ = validate_runtime(runtime)
    hashes = read_json(root / "source-hashes.json")
    require(all(re.fullmatch("[0-9a-f]{64}", hashes.get(f"experiments/fsdp2_ce/{name}", ""))
                for name in ("bench.py", "modal_run.py", "analyze_trace.py")), "Missing source provenance")
    config = read_json(root / "check-default" / "rank0.json")["config"]
    require(all(config[key] == value for key, value in XL_CONFIG.items()) and
            re.fullmatch("[0-9a-f]{64}", config["input_sha256"]), "Unexpected XL workload configuration")

    expected_order = [f"check-{variant}" for variant in VARIANTS]
    expected_order += [f"bench-{index}-{variant}" for index, order in enumerate(ROTATIONS) for variant in order]
    observed_order = [case["label"] for case in run["cases"][:len(expected_order)]]
    require(observed_order == expected_order, "Missing or misordered launcher parity and benchmark cases")
    reference, reference_shapes = check_case(root, "check-default", "default", runtime, config)
    checks = {"default": reference}
    for variant in VARIANTS[1:]:
        checks[variant], _ = check_case(root, f"check-{variant}", variant, runtime, config,
                                       reference_shapes=reference_shapes)
    rotations = []
    for index, order in enumerate(ROTATIONS):
        cases = {variant: bench_case(root, f"bench-{index}-{variant}", variant, runtime, config) for variant in order}
        baseline = cases["default"]["timing"]["median_ms"]
        for case in cases.values():
            case["median_ratio_to_default"] = case["timing"]["median_ms"] / baseline
        ce_ratio = cases["zero"]["timing"]["median_ms"] / cases["symm"]["timing"]["median_ms"]
        rotations.append({"rotation": index, "execution_order": order, "cases": cases,
                          "zero_to_symm_median_ratio": ce_ratio})

    variants = {}
    for variant in VARIANTS:
        cases = [rotation["cases"][variant] for rotation in rotations]
        memory = [row for case in cases for row in case["memory_by_rank"]]
        nvml = [row["nvml"]["observed_max_device_bytes"] for row in memory]
        variants[variant] = {
            "timing": distribution([step["max_rank_ms"] for case in cases for step in case["steps"]]),
            "slowest_rank_counts": dict(Counter(step["slowest_rank"] for case in cases for step in case["steps"])),
            "rotation_median_ratios_to_default": [case["median_ratio_to_default"] for case in cases],
            **allocator_peaks(memory),
            "observed_max_device_bytes": max((value for value in nvml if value is not None), default=None),
            "nvml_rank_runs_sampled": sum(value is not None for value in nvml),
            "nvml_rank_runs_expected": len(nvml),
        }

    return {
        "run_id": run["run_id"], "launcher_status": run["status"], "launcher_error": run.get("error"),
        "gpu_model": gpu_model, "runtime": runtime, "source_hashes": hashes, "config": config, "parity": checks,
        "timing_scope": "Wall time from zero_grad through forward, cross-entropy, backward, AdamW, and device synchronize. "
                        "Maximum of the two rank durations per step, without profiling. The pre-step barrier is excluded.",
        "memory_scope": "Allocator maxima cover both ranks and all timed steps. NVML is a sampled device-wide maximum "
                        "from separate three-step passes and may miss the true peak.",
        "variants": variants, "rotations": rotations,
        "rotation_zero_to_symm_median_ratios": [row["zero_to_symm_median_ratio"] for row in rotations],
        **wait_followup(root, run, runtime, config, reference_shapes),
    }


def as_markdown(result):
    lines = [
        f"# {result['run_id']}", "",
        "Full-run launcher status: " + result["launcher_status"], "",
        f"Hardware and runtime. 2× {result['gpu_model']}, PyTorch {result['runtime']['torch']}, "
        f"CUDA {result['runtime']['cuda']}, NCCL {'.'.join(map(str, result['runtime']['nccl']))}.", "",
        "| Variant | Steps | Mean ± SD (ms) | Median (ms) | P95 (ms) | Allocated (GiB) | Reserved (GiB) | NVML observed (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in result["variants"].items():
        timing = row["timing"]
        nvml = "unavailable" if row["observed_max_device_bytes"] is None else (
            f"{row['observed_max_device_bytes'] / GIB:.3f} "
            f"({row['nvml_rank_runs_sampled']}/{row['nvml_rank_runs_expected']} rank runs)"
        )
        lines.append(
            f"| {variant} | {timing['count']} | {timing['mean_ms']:.3f} ± {timing['stdev_ms']:.3f} | "
            f"{timing['median_ms']:.3f} | {timing['p95_ms']:.3f} | "
            f"{row['peak_allocated_bytes'] / GIB:.3f} | {row['peak_reserved_bytes'] / GIB:.3f} | {nvml} |"
        )
    lines += [
        "", "| Rotation | Execution order | Symmetric / default median | Zero-CTA / default median | Zero-CTA / symmetric median |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in result["rotations"]:
        cases = row["cases"]
        lines.append(f"| {row['rotation'] + 1} | {' → '.join(row['execution_order'])} | "
                     f"{cases['symm']['median_ratio_to_default']:.4f}× | {cases['zero']['median_ratio_to_default']:.4f}× | "
                     f"{row['zero_to_symm_median_ratio']:.4f}× |")
    lines += ["", "Parity passed for per-rank loss, full gradients, and post-AdamW parameters in all three variants."]
    diagnostic = result.get("wait_diagnostic", {})
    prefetch = diagnostic.get("prefetch")
    if prefetch and prefetch["status"] == "verified":
        lines += ["", "## Forward prefetch follow-up", "",
                  "Both cases use symmetric memory and NCCL zero-CTA. The explicit prefetch case passed parity against the original FSDP2 reference.", "",
                  "| Forward prefetch | Steps | Median (ms) | Allocated (GiB) | Reserved (GiB) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for strategy, case in prefetch["cases"].items():
            timing = case["timing"]
            lines.append(f"| {strategy} | {timing['count']} | {timing['median_ms']:.3f} | "
                         f"{case['peak_allocated_bytes'] / GIB:.3f} | {case['peak_reserved_bytes'] / GIB:.3f} |")
        lines += ["", f"Explicit next-layer prefetch changed the median by {prefetch['median_delta_ms']:+.3f} ms "
                  f"({prefetch['median_ratio']:.4f}×). Decision evidence: `{diagnostic['path']}`."]
    elif "error" in diagnostic:
        lines += ["", f"The follow-up could not be verified from `{diagnostic['path']}`. {diagnostic['error']}"]
    lines += ["", result["timing_scope"], "", result["memory_scope"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="Write to a file instead of standard output")
    args = parser.parse_args()
    try:
        result = summarize(args.run_dir)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n" if args.format == "json" else as_markdown(result)
        if args.output is None:
            print(content, end="")
        else:
            args.output.write_text(content)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"Invalid run evidence: {error}\n")


if __name__ == "__main__":
    main()
