"""Independently verify a downloaded six-case FSDP2 smoke run on two B200s or two B300s."""

import argparse
import json
from pathlib import Path
import re
import sqlite3

from analyze_trace import analyze
from summarize import TOLERANCES, VARIANTS, is_finite, read_json, require, tensor_check, validate_runtime


NSYS_VERSION = "2026.5.1.161"
SMOKE_MODEL = {"parameters": 4_720_896, "d_model": 256, "d_ff": 1024, "num_layers": 4,
               "num_heads": 4, "vocab_size": 1024, "context_length": 128,
               "global_batch": 4, "dtype": "float32", "activation_checkpointing": False,
               "tf32": False, "optimizer": "torch.optim.AdamW", "adamw_lr": 1e-3,
               "adamw_foreach": False, "adamw_fused": False,
               "reshard_after_forward": {"block": True, "root": False}}


def nonempty(path):
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty artifact {path}")
    return path


def successful(status, label):
    require(status["returncode"] == 0 and status["timed_out"] is False, f"{label} did not finish successfully")


def topology_rows(path):
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        parts = re.sub(r"\x1b\[[0-9;]*m", "", line).split()
        if len(parts) >= 3 and parts[0] in {"GPU0", "GPU1"} and parts[0] not in rows:
            rows[parts[0]] = parts[1:3]
    return rows


def topology(root, runtime, gpu_model):
    query = nonempty(root / "nvidia-smi.txt").read_text()
    gpu_uuids = re.findall(r"(?m)^\s*GPU UUID\s*:\s*(GPU-[a-f0-9-]+)", query, re.IGNORECASE)
    nvlink_status = root / "nvlink-status.txt"
    status_text = nvlink_status.read_text() if nvlink_status.is_file() else ""
    if not gpu_uuids:
        gpu_uuids = re.findall(r"(?m)^GPU [01]:.*\(UUID:\s*(GPU-[a-f0-9-]+)\)", status_text, re.IGNORECASE)
    require(len(gpu_uuids) == 2 and len(set(gpu_uuids)) == 2, "Expected two distinct physical GPU UUIDs")
    driver = re.search(r"(?m)^\s*Driver Version\s*:\s*(\S+)", query)
    require(driver is not None, "Missing driver version in nvidia-smi.txt")
    evidence = []
    matrix = topology_rows(root / "topology.txt")
    matrix_pair = (matrix.get("GPU0", ["", ""])[1], matrix.get("GPU1", ["", ""])[0])
    if all(re.fullmatch(r"NV\d+", cell) for cell in matrix_pair):
        evidence.append({"artifact": "topology.txt", "gpu0_to_gpu1": matrix_pair[0], "gpu1_to_gpu0": matrix_pair[1]})
    nvlink_p2p = topology_rows(root / "p2p-nvlink.txt")
    peer_pair = (nvlink_p2p.get("GPU0", ["", ""])[1], nvlink_p2p.get("GPU1", ["", ""])[0])
    if peer_pair == ("OK", "OK"):
        evidence.append({"artifact": "p2p-nvlink.txt", "gpu0_to_gpu1": "OK", "gpu1_to_gpu0": "OK"})

    comm = re.compile(r"\brank ([01]) nRanks 2 nNodes 1 localRanks 2 localRank [01]\b")
    path_pattern = r"GPU/[a-f0-9]+-[a-f0-9]+ \(\d+/[\d.]+/NVL\)"
    single_node_logs = []
    for variant in VARIANTS:
        log = root / f"trace-{variant}" / "worker.log"
        lines = nonempty(log).read_text().splitlines()
        if {match[1] for line in lines if (match := comm.search(line))} == {"0", "1"}:
            single_node_logs.append(str(log.relative_to(root)))
        paths = [line.strip() for line in lines if re.search(path_pattern, line, re.IGNORECASE)]
        if paths:
            evidence.append({"artifact": str(log.relative_to(root)), "nccl_gpu_nvlink_paths": list(dict.fromkeys(paths))})
    return {
        "gpu_uuids": gpu_uuids, "driver_version": driver[1],
        "bidirectional_cuda_peer_access": runtime["cuda_peer_access"],
        "single_node_nccl_log_artifacts": single_node_logs,
        "active_nvlink_status_artifact": "nvlink-status.txt" if re.search(r"Link \d+: [\d.]+ GB/s", status_text) else None,
        "nvlink_domain_confirmed": bool(evidence), "nvlink_evidence": evidence,
        "scope": f"Two local {gpu_model} ranks exercised same-node NCCL copy-engine all-gather with CUDA P2P. "
                 "Pairwise NVLink is independently confirmed only by an NVLink-specific peer matrix or a GPU-to-GPU "
                 "NVL path in NCCL logs. Active links alone do not identify the peer or measure NVLink traffic.",
    }


def check_parity(label, variant, rows, config, reference):
    check = rows[0]["check"]
    recording = variant == "default"
    require(all(row["passed"] is True for row in rows) and check["passed"] is True and
            check["recorded_reference"] is recording, f"{label}: parity failed")
    require(check["tolerances"] == TOLERANCES, f"{label}: wrong parity tolerances")
    losses = check["rank_losses"]
    loss = check["loss_check"]
    require(len(losses) == 2 and all(is_finite(value) for value in losses) and loss["finite"] is True,
            f"{label}: missing or nonfinite rank loss")
    if not recording:
        require(loss["mismatched_elements"] == 0 and is_finite(loss["max_abs_error"]) and loss["max_abs_error"] >= 0,
                f"{label}: loss differs from baseline")
    grad_shapes, gradients = tensor_check(check["gradients"], recording, config["parameters"], f"{label}/gradients")
    param_shapes, parameters = tensor_check(check["parameters"], recording, config["parameters"], f"{label}/parameters")
    require(grad_shapes == param_shapes and (recording or param_shapes == reference), f"{label}: tensor inventory mismatch")
    return {"passed": True, "rank_losses": losses, "max_abs_loss_error": loss.get("max_abs_error"),
            "gradients": gradients, "parameters": parameters}, param_shapes


def check_trace(folder, variant, rows, status, *, recover_analysis=False):
    successful(status["export"], f"{folder.name}/export")
    if not recover_analysis:
        successful(status["analysis"], f"{folder.name}/analysis")
    nonempty(folder / "profile.nsys-rep")
    sqlite_path = nonempty(folder / "profile.sqlite")
    # analyze() opens SQLite with mode=ro. Ignore saved trace.json conclusions here.
    trace = analyze(sqlite_path, folder, expected_ranks=2)
    require(trace["two_rank_gpu_coverage"], f"{folder.name}: both ranks must have distinct GPU activity")
    for rank, row in enumerate(rows):
        samples = row["samples"]
        waits = row["waits"]
        require(row["profiled"] is True and row["warmup"] == 5 and len(samples) == 1 and
                samples[0]["rank"] == rank and samples[0]["step"] == 0 and
                is_finite(samples[0]["duration_ms"]) and samples[0]["duration_ms"] > 0 and
                is_finite(samples[0]["loss"]) and bool(waits), f"{folder.name}: incomplete rank {rank} samples")
        evidence = trace["ranks"][str(rank)]
        require(len(evidence["steps"]) == 1 and evidence["steps"][0]["step"] == 0 and
                evidence["unmatched_wait_records"] == 0 and evidence["gather_routes"],
                f"{folder.name}: incomplete trace/wait correspondence for rank {rank}")
    if variant == "zero":
        require(trace["copy_engine_confirmed_all_ranks"], f"{folder.name}: correlated CE AllGather and GPU copy missing")
    else:
        require(all(not ({"copy_engine", "mixed"} & set(row["gather_routes"])) for row in trace["ranks"].values()),
                f"{folder.name}: unexpected CE activity in a default-CTA control")
    return {"sqlite": str(sqlite_path), "copy_engine_confirmed_all_ranks": trace["copy_engine_confirmed_all_ranks"],
            "copy_engine_only_all_gathers": trace["copy_engine_only_all_gathers"],
            "ranks": {rank: {key: row[key] for key in ("process_id", "devices", "gather_routes", "exposed_wait_ms")}
                      for rank, row in trace["ranks"].items()}}


def audit(root, *, recover_analysis=False):
    runtime = read_json(root / "runtime.json")
    gpu_model, stack = validate_runtime(runtime)
    successful(runtime["diagnostics"]["nsys-version"], "Nsight version check")
    nsys = nonempty(root / "nsys-version.txt").read_text().strip()
    require(re.search(rf"\bversion {re.escape(NSYS_VERSION)}(?:-|\s|$)", nsys), f"Unexpected Nsight release: {nsys}")
    run = read_json(root / "summary.json")
    progress = read_json(root / "progress.json")
    labels = [f"{mode}-{variant}" for mode in ("check", "trace") for variant in VARIANTS]
    require(run["stage"] == "smoke" and progress == run["cases"] and
            [case["label"] for case in run["cases"]] == labels, "Run did not complete the six smoke cases")
    recovered = None
    if run["status"] != "ok":
        require(recover_analysis, f"Launcher reported failure: {run.get('error')}")
        last = run["cases"][-1]
        expected_error = "Trace evidence did not pass for trace-zero"
        require(run["status"] == "failed" and run.get("error") == expected_error and
                last["status"] == "failed" and last.get("error") == expected_error and
                last["analysis"]["returncode"] == 2 and last["analysis"]["timed_out"] is False,
                "Only the original trace-zero analyzer rejection can be recovered offline")
        recovered = {"case": last["label"], "original_status": last["status"], "original_error": last["error"],
                     "original_analysis_returncode": last["analysis"]["returncode"],
                     "original_analysis_log": str(root / last["label"] / "analysis.log"),
                     "verification": "Independently reanalyzed the original SQLite and confirmed CE AllGather plus GPU copies on both ranks."}

    config = read_json(root / "check-default" / "rank0.json")["config"]
    require(all(config.get(key) == value for key, value in SMOKE_MODEL.items()) and
            re.fullmatch(r"[a-f0-9]{64}", config["input_sha256"]), "Unexpected smoke workload")
    parity, traces = {}, {}
    ref = None
    for case in run["cases"]:
        label = case["label"]
        mode, variant = label.split("-", 1)
        folder = root / label
        recover_case = recovered is not None and label == recovered["case"]
        expected_status = "failed" if recover_case else "ok"
        require(case == read_json(folder / "status.json") and case["status"] == expected_status and
                case["mode"] == mode and case["variant"] == variant and case["workload"] == "smoke",
                f"{label}: case metadata disagrees")
        successful(case["worker"], label)
        rows = [read_json(folder / f"rank{rank}.json") for rank in (0, 1)]
        for rank, row in enumerate(rows):
            expected = stack | {"rank": rank, "gpu": runtime["gpu_names"][rank], "world_size": 2, "mode": mode,
                                "workload": "smoke", "variant": variant, "config": config, "seed": 2026,
                                "forward_prefetch": "default",
                                "nccl_cta_policy": "ZERO" if variant == "zero" else "DEFAULT"}
            require(all(row.get(key) == value for key, value in expected.items()), f"{label}: invalid rank {rank} metadata")
        if mode == "check":
            parity[variant], ref = check_parity(label, variant, rows, config, ref)
        else:
            traces[variant] = check_trace(folder, variant, rows, case, recover_analysis=recover_case)

    device_maps = [tuple(row["devices"][0] for row in case["ranks"].values()) for case in traces.values()]
    require(all(mapping == device_maps[0] for mapping in device_maps), "Rank-to-device mapping differs across traces")
    return {"status": "passed", "run_id": run["run_id"], "gpu_model": gpu_model, "stack": stack | {"nsys": nsys},
            "launcher_status": run["status"], "launcher_error": run.get("error"), "recovered_analysis": recovered,
            "config": config, "topology": topology(root, runtime, gpu_model), "parity": parity, "traces": traces}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Downloaded Modal smoke artifact directory")
    parser.add_argument("--recover-analysis", action="store_true",
                        help="Reanalyze a trace-zero analyzer rejection after a successful worker and export")
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.run_directory, recover_analysis=args.recover_analysis), indent=2, allow_nan=False))
    except (KeyError, IndexError, TypeError, ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Smoke verification failed: {exc}\n")


if __name__ == "__main__":
    main()
