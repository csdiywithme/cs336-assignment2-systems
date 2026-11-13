"""Run the FSDP2 all-gather comparison on one Modal dual-GPU container."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import modal


REMOTE = Path("/workspace")
EXPERIMENT = REMOTE / "experiments" / "fsdp2_ce"
HERE = Path(__file__).resolve().parent if modal.is_local() else EXPERIMENT
ROOT = HERE.parents[1]
VOLUME_NAME = "cs336-assignment2-20260914"
GPU_MODEL = os.environ.get("FSDP2_CE_GPU_MODEL", "B200")
CUDA_WHEEL = {"B200": "cu130", "B300": "cu132"}[GPU_MODEL]
NSYS_PACKAGE = (
    "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
    "NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb"
)
VARIANTS = ("default", "symm", "zero")
STAGES = {"smoke", "full", "prefetch", "micro", "route"}

app = modal.App("cs336-fsdp2-copy-engine")
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "wget")
    .pip_install("torch==2.12.1", index_url=f"https://download.pytorch.org/whl/{CUDA_WHEEL}")
    .pip_install("einops==0.8.2", "einx==0.4.3", "jaxtyping==0.3.9", "nvidia-ml-py==13.580.82")
    .run_commands(
        f"wget --no-verbose {NSYS_PACKAGE} -O /tmp/nsys.deb",
        "apt-get install -y --no-install-recommends /tmp/nsys.deb",
        "rm /tmp/nsys.deb",
        "nsys --version",
    )
    .env({
        "PYTHONPATH": "/workspace:/workspace/cs336-basics",
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "NSYS_NVTX_PROFILER_REGISTER_ONLY": "0",
        "FSDP2_CE_GPU_MODEL": GPU_MODEL,
    })
)
if modal.is_local():
    image = (
        image.add_local_dir(HERE, str(EXPERIMENT), ignore=["**/__pycache__/**", "**/*.pyc"])
        .add_local_dir(ROOT / "cs336_systems", "/workspace/cs336_systems", ignore=["**/__pycache__/**"])
        .add_local_dir(ROOT / "cs336-basics" / "cs336_basics", "/workspace/cs336-basics/cs336_basics",
                       ignore=["**/__pycache__/**"])
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def logged_command(command, folder, log_name, timeout, extra_env=None):
    env = os.environ.copy()
    env.update(extra_env or {})
    started = perf_counter()
    timed_out = False
    with (folder / log_name).open("w") as log:
        process = subprocess.Popen(command, cwd=REMOTE, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return {
        "command": command, "environment": extra_env or {}, "returncode": process.returncode,
        "timed_out": timed_out, "elapsed_s": perf_counter() - started,
    }


def run_case(root, reference, workload, variant, mode, *, label, repeats, cuda_trace,
             forward_prefetch="default", extra_env=None):
    folder = root / label
    folder.mkdir()
    worker = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(EXPERIMENT / ("microbench.py" if workload == "micro" else "bench.py")),
        "--variant", variant, "--output", str(folder), "--warmup", "5", "--steps", str(repeats),
    ]
    if workload == "micro":
        if mode == "trace":
            worker.append("--trace")
    else:
        worker += ["--workload", workload, "--mode", mode, "--forward-prefetch", forward_prefetch]
    if mode == "check":
        worker += ["--reference", str(reference)]
    env = dict(extra_env or {})
    if mode == "trace":
        capture = "fsdp2.microbench" if workload == "micro" else "fsdp2.measurement"
        worker = [
            "nsys", "profile", f"--trace={cuda_trace},nvtx,nccl,cublas",
            "--nccl-trace=api,rt,gpu,ce-coll,ce-sync,ce-batch",
            "--cuda-event-trace=true", "--sample=none", "--cpuctxsw=none",
            "--capture-range=nvtx", f"--nvtx-capture={capture}", "--capture-range-end=none",
            "-o", str(folder / "profile"), *worker,
        ]
        env.update({
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,REG,TUNING",
            "NCCL_TOPO_DUMP_FILE": str(folder / "nccl-topology.xml"),
        })
    result = {"workload": workload, "mode": mode, "variant": variant, "label": label}
    try:
        result["worker"] = logged_command(worker, folder, "worker.log", 300, env)
        if result["worker"]["returncode"]:
            raise RuntimeError(f"Worker failed for {label}")
        for rank in (0, 1):
            if not (folder / f"rank{rank}.json").is_file():
                raise RuntimeError(f"Missing rank {rank} results for {label}")
        if mode == "trace":
            report = folder / "profile.nsys-rep"
            sqlite = folder / "profile.sqlite"
            export = ["nsys", "export", "--type", "sqlite", "--include-json=true",
                      "--output", str(sqlite), str(report)]
            result["export"] = logged_command(export, folder, "export.log", 100)
            if result["export"]["returncode"] or not sqlite.is_file():
                raise RuntimeError(f"Missing SQLite trace for {label}")
            analyzer = "analyze_micro_trace.py" if workload == "micro" else "analyze_trace.py"
            analyze = [sys.executable, str(EXPERIMENT / analyzer), str(sqlite), "--output", str(folder / "trace.json")]
            baseline_case = "trace-zero-default" if label == "trace-zero-next" else "trace-default"
            baseline = root / baseline_case / "trace.json"
            if workload != "micro" and baseline.is_file() and baseline.parent != folder:
                analyze += ["--baseline", str(baseline)]
            if variant == "zero":
                analyze.append("--require-ce")
            result["analysis"] = logged_command(analyze, folder, "analysis.log", 100)
            if result["analysis"]["returncode"] or not (folder / "trace.json").is_file():
                raise RuntimeError(f"Trace evidence did not pass for {label}")
        result["status"] = "ok"
    except (OSError, RuntimeError) as error:
        result["status"] = "failed"
        result["error"] = str(error)
    write_json(folder / "status.json", result)
    if result["status"] == "failed" and (folder / "worker.log").exists():
        with (folder / "worker.log").open("rb") as log:
            log.seek(-min(6000, (folder / "worker.log").stat().st_size), os.SEEK_END)
            print(log.read().decode(errors="replace"), flush=True)
    return result


def plan(stage):
    if stage == "smoke":
        for variant in VARIANTS:
            yield "smoke", variant, "check", f"check-{variant}", 1, "default"
        for variant in VARIANTS:
            yield "smoke", variant, "trace", f"trace-{variant}", 1, "default"
    elif stage == "full":
        for variant in VARIANTS:
            yield "xl", variant, "check", f"check-{variant}", 1, "default"
        for rotation in range(len(VARIANTS)):
            for variant in VARIANTS[rotation:] + VARIANTS[:rotation]:
                yield "xl", variant, "bench", f"bench-{rotation}-{variant}", 12, "default"
        for variant in VARIANTS:
            yield "xl", variant, "trace", f"trace-{variant}", 2, "default"
    elif stage == "prefetch":
        yield "xl", "default", "check", "check-default", 1, "default"
        yield "xl", "zero", "check", "check-zero-default", 1, "default"
        yield from prefetch_cases()
    elif stage == "micro":
        yield from micro_cases()
    elif stage == "route":
        for variant in VARIANTS:
            yield "xl", variant, "check", f"check-{variant}", 1, "default"
        for rotation, variants in enumerate((("symm", "zero"), ("zero", "symm"))):
            for variant in variants:
                yield "xl", variant, "bench", f"bench-{rotation}-{variant}", 12, "default"
        for variant in ("symm", "zero"):
            yield "xl", variant, "trace", f"trace-{variant}", 2, "default"


def prefetch_cases():
    yield "xl", "zero", "check", "check-zero-next", 1, "next"
    for prefetch in ("default", "next"):
        yield "xl", "zero", "bench", f"bench-zero-{prefetch}", 12, prefetch
        yield "xl", "zero", "trace", f"trace-zero-{prefetch}", 2, prefetch


def micro_cases():
    for variant in ("symm", "zero"):
        yield "micro", variant, "bench", f"micro-bench-{variant}", 15, "default"
        yield "micro", variant, "trace", f"micro-trace-{variant}", 3, "default"


def wait_diagnostic(root):
    """Choose a control from exposed waits in the CE-enabled trace."""
    trace = json.loads((root / "trace-zero" / "trace.json").read_text())
    rows = []
    for step, profile in trace["slowest_profiled_rank_by_step"].items():
        rank = profile["rank"]
        current = next(row for row in trace["ranks"][str(rank)]["steps"] if row["step"] == int(step))
        for phase in ("forward", "backward"):
            measured = [row["measured_wait"] for row in current["layers"] if row["phase"] == phase
                        and (phase != "forward" or row["layer"] not in {"root", "00"})
                        and row["measured_wait"] is not None]
            if measured:
                rows.append({"rank": rank, "step": int(step), "phase": phase, "measured_blocks": len(measured),
                             "exposed_wait_ms": sum(row["exposed_wait_ms"] for row in measured)})
    medians = {}
    for phase in ("forward", "backward"):
        durations = [row["exposed_wait_ms"] for row in rows if row["phase"] == phase]
        medians[phase] = statistics.median(durations) if durations else None
    threshold_ms = 1.0
    followup = None
    if (medians["forward"] or 0) >= threshold_ms:
        followup = "prefetch"
    elif (medians["backward"] or 0) >= threshold_ms:
        followup = "micro"
    return {"samples": rows, "median_wait_ms": medians, "threshold_ms": threshold_ms, "followup": followup}


def run_plan(stage, root):
    yield from plan(stage)
    if stage == "full":
        decision = wait_diagnostic(root)
        write_json(root / "wait-diagnostic.json", decision)
        if decision["followup"] == "prefetch":
            yield from prefetch_cases()
        elif decision["followup"] == "micro":
            yield from micro_cases()


def validate_runtime(root):
    import torch
    from torch.distributed.fsdp import FSDPModule

    diagnostics = {}
    for name, command in {
        "nsys-version": ["nsys", "--version"],
        "nvidia-smi": ["nvidia-smi", "-q"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "p2p-nvlink": ["nvidia-smi", "topo", "-p2p", "n"],
        "nvlink-status": ["nvidia-smi", "nvlink", "--status"],
        "pip-freeze": [sys.executable, "-m", "pip", "freeze"],
    }.items():
        diagnostics[name] = logged_command(command, root, f"{name}.txt", 25)
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    peers = [torch.cuda.can_device_access_peer(0, 1), torch.cuda.can_device_access_peer(1, 0)] if len(gpu_names) == 2 else []
    runtime = {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(), "gpu_model": GPU_MODEL, "gpu_names": gpu_names, "cuda_peer_access": peers,
        "symm_api": hasattr(FSDPModule, "set_symm_mem_for_comm"), "diagnostics": diagnostics,
    }
    write_json(root / "runtime.json", runtime)
    if not (len(gpu_names) == 2 and gpu_names[0] == gpu_names[1]
            and all(re.search(rf"\b{GPU_MODEL}\b", name) for name in gpu_names) and peers == [True, True]):
        raise RuntimeError(f"Expected two matching {GPU_MODEL} GPUs with bidirectional P2P")
    if not runtime["symm_api"] or tuple(runtime["nccl"]) < (2, 29):
        raise RuntimeError("FSDP2 symmetric memory or NCCL advanced CE tracing unavailable")
    if diagnostics["nsys-version"]["returncode"]:
        raise RuntimeError("Nsight Systems unavailable")


@app.function(image=image, gpu=f"{GPU_MODEL}:2", cpu=8, memory=65536,
              timeout=1800, startup_timeout=600, retries=0, single_use_containers=True,
              volumes={"/data": volume})
def experiment(run_id: str, stage: str, cuda_trace: str, hashes: dict[str, str]):
    if not re.fullmatch(r"fsdp2-ce-[\w-]+", run_id):
        raise ValueError("Invalid run ID")
    if stage not in STAGES or cuda_trace not in {"cuda", "cuda-sw"}:
        raise ValueError("Invalid experiment stage or CUDA tracing backend")
    volume.reload()
    root = Path("/data") / run_id
    root.mkdir(exist_ok=False)
    reference = Path("/tmp") / run_id
    result = {"run_id": run_id, "function_call_id": modal.current_function_call_id(), "stage": stage,
              "gpu_model": GPU_MODEL,
              "cuda_trace": cuda_trace, "cases": [], "status": "failed"}
    started = perf_counter()
    try:
        write_json(root / "source-hashes.json", hashes)
        for name, expected in hashes.items():
            data = (REMOTE / name).read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Uploaded source hash mismatch: {name}")
            snapshot = root / "source" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data)
        validate_runtime(root)
        volume.commit()
        # Large baseline tensors are temporary. Persist only numeric comparisons.
        reference.mkdir()
        extra_env = {"NCCL_SYM_CE_THRESHOLD": "1073741824"} if stage == "route" else None
        for workload, variant, mode, label, repeats, prefetch in run_plan(stage, root):
            print(f"Starting {run_id}/{label}", flush=True)
            case = run_case(root, reference, workload, variant, mode, label=label, repeats=repeats,
                            cuda_trace=cuda_trace, forward_prefetch=prefetch, extra_env=extra_env)
            result["cases"].append(case)
            write_json(root / "progress.json", result["cases"])
            volume.commit()
            print(json.dumps(case), flush=True)
            if case["status"] != "ok":
                raise RuntimeError(case["error"])
        result["status"] = "ok"
    except (OSError, RuntimeError) as error:
        result["error"] = str(error)
    finally:
        result["elapsed_s"] = perf_counter() - started
        write_json(root / "summary.json", result)
        volume.commit()
    return result


@app.function(image=image, timeout=120, startup_timeout=600)
def image_check():
    import torch
    from torch.distributed.fsdp import FSDPModule

    version = subprocess.check_output(["nsys", "--version"], text=True).strip()
    help_text = subprocess.check_output(["nsys", "profile", "--help"], stderr=subprocess.STDOUT, text=True)
    for feature in ("ce-coll", "--nccl-trace", "--cuda-event-trace", "cuda-sw"):
        if feature not in help_text:
            raise RuntimeError(f"Expected Nsight profiler feature: {feature}")
    if not hasattr(FSDPModule, "set_symm_mem_for_comm"):
        raise RuntimeError("Pinned PyTorch lacks the FSDP2 symmetric-memory API")
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
            "nsys": version}


@app.local_entrypoint()
def main(stage: str = "smoke", cuda_trace: str = "cuda-sw"):
    if stage == "image":
        print(json.dumps(image_check.remote(), indent=2))
        return
    if stage not in STAGES or cuda_trace not in {"cuda", "cuda-sw"}:
        raise ValueError("Choose image/smoke/full/prefetch/micro/route and cuda/cuda-sw")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"fsdp2-ce-{stage}-{GPU_MODEL.lower()}-{stamp}-{uuid4().hex[:6]}"
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (HERE, ROOT / "cs336_systems", ROOT / "cs336-basics" / "cs336_basics")
        for path in sorted(folder.rglob("*.py"))
    }
    print(f"Run ID {run_id} on volume {VOLUME_NAME}", flush=True)
    worker = experiment.with_options(cpu=4, memory=16384) if stage == "smoke" else experiment
    call = worker.spawn(run_id, stage, cuda_trace, hashes)
    receipt = {"run_id": run_id, "function_call_id": call.object_id, "stage": stage, "cuda_trace": cuda_trace,
               "gpu_model": GPU_MODEL, "profile": os.environ.get("MODAL_PROFILE"), "volume": VOLUME_NAME}
    dispatch = ROOT / "runs" / "dispatch" / run_id
    dispatch.mkdir(parents=True, exist_ok=False)
    write_json(dispatch / "dispatch.json", receipt)
    print(json.dumps(receipt), flush=True)
    result = call.get()
    write_json(dispatch / "summary.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise RuntimeError(f"{run_id}: {result.get('error', 'failed')}")
