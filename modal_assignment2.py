"""Bounded, reproducible Assignment 2 jobs. Importing never launches a job.

Examples (from this directory):
  modal run --detach modal_assignment2.py --plan probe --gpus 1
  modal volume get cs336-assignment2-20260914 RUN_ID runs/cloud/RUN_ID
"""
from pathlib import Path
import datetime
import hashlib
import json
import os

import modal

ROOT = Path(__file__).resolve().parent
GPU_COUNT = int(os.environ.get("A2_GPUS", "1"))
TIMEOUT = int(os.environ.get("A2_TIMEOUT", "1800"))
if GPU_COUNT not in (1, 2, 4, 6) or not 60 <= TIMEOUT <= 3600:
    raise ValueError("Only 1/2/4/6 B200 and a 60–3600 second job are allowed")
app = modal.App("cs336-assignment2-20260914")
volume = modal.Volume.from_name("cs336-assignment2-20260914", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("einops==0.8.2", "einx==0.4.3", "jaxtyping==0.3.9", "numpy==2.4.4",
                 "psutil==7.2.2", "pytest==9.0.2", "pytest-timeout==2.4.0", "regex", "tqdm", "wandb")
)
if os.environ.get("A2_NSYS") == "1":
    image = (image.apt_install("wget", "gnupg", "ca-certificates")
             .add_local_file(ROOT / "scripts/nvidia-devtools.list", "/etc/apt/sources.list.d/nvidia-devtools.list", copy=True)
             .run_commands("wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub -O /tmp/nvidia-devtools.pub",
                           "gpg --dearmor --output /usr/share/keyrings/nvidia-devtools-keyring.gpg /tmp/nvidia-devtools.pub",
                           "apt-get update", "apt-get install -y --no-install-recommends nsight-systems-cli"))
image = (image
    .env({"PYTHONPATH": "/workspace/cs336-basics:/workspace", "OMP_NUM_THREADS": "8",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "cs336_systems", "/workspace/cs336_systems", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / "cs336-basics" / "cs336_basics", "/workspace/cs336-basics/cs336_basics", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / "tests", "/workspace/tests", ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, gpu=f"B200:{GPU_COUNT}", cpu=(8, 8), memory=(65536, 65536),
              timeout=TIMEOUT, startup_timeout=600, retries=0, max_containers=2,
              single_use_containers=True, volumes={"/data": volume})
def experiment(run_id: str, cases: list[dict], provenance: dict):
    import platform
    import subprocess
    import sys
    import time
    import traceback
    import torch
    import triton

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("unsafe run id")
    volume.reload()
    out = Path("/data") / run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "triton": triton.__version__, "gpu_count": torch.cuda.device_count(),
           "gpus": [str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())],
           "provenance": provenance, "cases": cases, "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
           "job_timeout_seconds": provenance.get("job_timeout_seconds", TIMEOUT),
           "estimate_usd_per_second": torch.cuda.device_count() * 0.001736 + 8 * 0.0000131 + 64 * 0.00000222}
    for command, name in [(["nvidia-smi", "-q"], "nvidia-smi.txt"),
                          (["nvidia-smi", "topo", "-m"], "topology.txt"),
                          (["nvidia-smi", "nvlink", "--status"], "nvlink-status.txt"),
                          ([sys.executable, "-m", "pip", "freeze"], "pip-freeze.txt")]:
        r = subprocess.run(command, capture_output=True, text=True)
        (out / name).write_text(r.stdout + r.stderr)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    # Copy the exact uploaded implementation; later local edits cannot change this evidence.
    import shutil
    shutil.copytree("/workspace/cs336_systems", out / "source", ignore=shutil.ignore_patterns("__pycache__"))
    volume.commit()
    results = []
    try:
        for index, case in enumerate(cases):
            case_dir = out / f"{index:03d}-{case['name']}"
            case_dir.mkdir()
            if case.get("pytest"):
                cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", *case["pytest"],
                       f"--junitxml={case_dir}/junit.xml"]
            else:
                cmd = [sys.executable, "-m", "cs336_systems.experiments", "--config", json.dumps(case),
                       "--output", str(case_dir)]
                if case.get("nsys"):
                    cmd = ["nsys", "profile", "--trace=cuda-sw,nvtx,osrt,cublas", "--sample=none", "--cpuctxsw=none",
                           "--capture-range=nvtx", "--nvtx-capture=measurement", "--capture-range-end=stop",
                           "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                           "--cuda-memory-usage=true", "--pytorch=autograd-nvtx",
                           "--export=sqlite", "-o", str(case_dir / "profile"), *cmd]
            before = time.time()
            print(f"CASE {index}: {case['name']}", flush=True)
            status = {"name": case["name"], "command": cmd, "started": before}
            with (case_dir / "stdout.log").open("w") as log:
                try:
                    process = subprocess.Popen(cmd, cwd="/workspace", stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        status["returncode"] = process.wait(timeout=case.get("timeout", 300))
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        status["status"] = "timeout"
                except Exception:
                    status["error"] = traceback.format_exc()
            status["seconds"] = time.time() - before
            status.setdefault("status", "ok" if status.get("returncode") == 0 else "failed")
            if case.get("nsys"):
                sqlite = case_dir / "profile.sqlite"
                status["trace_created"] = sqlite.exists()
                if sqlite.exists():
                    r = subprocess.run([sys.executable, "-m", "cs336_systems.nsight_summary", str(sqlite),
                                        "--output-dir", str(sqlite.parent)],
                                       cwd="/workspace", capture_output=True, text=True, timeout=120)
                    (case_dir / "trace-analysis.log").write_text(r.stdout + r.stderr)
            if (case_dir / "result.json").exists():
                status["result"] = json.loads((case_dir / "result.json").read_text())
            (case_dir / "status.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
            print((case_dir / "stdout.log").read_text()[-8000:], flush=True)
            results.append(status)
            (out / "progress.json").write_text(json.dumps(results, indent=2))
            volume.commit()
    finally:
        elapsed = time.time() - started
        summary = {"run_id": run_id, "seconds": elapsed, "cases": results,
                   "estimated_compute_usd": elapsed * env["estimate_usd_per_second"],
                   "cost_note": "Runtime estimate, not invoice; excludes startup/build/storage."}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        inventory = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size}
                     for p in out.rglob("*") if p.is_file()]
        (out / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2))
        volume.commit()
    return summary


@app.function(image=modal.Image.debian_slim(), timeout=120, retries=0, volumes={"/data": volume})
def read_small_results(run_id: str):
    """RPC fallback for networks unable to reach Volume block-download URLs."""
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("unsafe run id")
    root = Path("/data") / run_id
    volume.reload()
    return {str(p.relative_to(root)): p.read_text() for p in root.rglob("*")
            if p.is_file() and not any("cache" in part for part in p.relative_to(root).parts)
            and p.suffix in (".json", ".log", ".xml", ".txt", ".py") and p.stat().st_size < 500000}


@app.local_entrypoint()
def main(plan: str = "probe", gpus: int = 1, tag: str = "", start: int = 0, end: int = 0, read_run: str = ""):
    from cs336_systems.experiment_plans import make_plan
    if read_run:
        for requested_run in read_run.split(","):
            files = read_small_results.remote(requested_run)
            local = ROOT / "runs/cloud" / requested_run / datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ-rpc")
            local.mkdir(parents=True, exist_ok=False)
            for relative, content in files.items():
                path = local / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            (local / "download_manifest.json").write_text(json.dumps({name: hashlib.sha256(content.encode()).hexdigest()
                                                                      for name, content in files.items()}, indent=2))
            print(f"Downloaded {len(files)} text artifacts to {local}")
        return
    if gpus != GPU_COUNT:
        raise ValueError("Set A2_GPUS before launching so allocation matches --gpus")
    cases = make_plan(plan, gpus)[start:end or None]
    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + plan + ("-" + tag if tag else "")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("cs336_systems", "tests", "cs336-basics/cs336_basics")
              for p in (ROOT / folder).rglob("*.py")}
    provenance = {"sha256": hashes, "gpu": f"B200:{GPU_COUNT}", "job_timeout_seconds": TIMEOUT, "modal_sdk": modal.__version__,
                  "handout_sha256": hashlib.sha256((ROOT / "cs336_assignment2_systems.pdf").read_bytes()).hexdigest()}
    local = ROOT / "runs" / "dispatch" / run_id
    local.mkdir(parents=True, exist_ok=False)
    (local / "request.json").write_text(json.dumps({"cases": cases, "provenance": provenance}, indent=2))
    call = experiment.spawn(run_id, cases, provenance)
    record = {"run_id": run_id, "call_id": call.object_id, "gpu_count": GPU_COUNT,
              "timeout_seconds": TIMEOUT, "plan": plan}
    (local / "dispatch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
