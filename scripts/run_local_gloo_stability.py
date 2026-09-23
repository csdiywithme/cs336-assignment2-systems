"""Five independent, bounded original CPU/Gloo test runs; never starts cloud jobs."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/local/final-gloo-stability"
TESTS = ["tests/test_ddp.py", "tests/test_fsdp.py", "tests/test_sharded_optimizer.py"]


def source_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("cs336_systems", "tests") for path in sorted((ROOT / folder).glob("*.py"))}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Fresh output directory; existing directories are never overwritten.")
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
                "python_version": sys.version, "tests": TESTS, "source_sha256_before": before,
                "backend": "CPU/Gloo", "gpu_used": False, "cloud_used": False,
                "rounds_requested": 5, "per_round_timeout_seconds": 120,
                "environment_overrides": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                           "PYTHONPATH": f"{ROOT / 'cs336-basics'}:{ROOT}"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = {**os.environ, **manifest["environment_overrides"]}
    results = {"backend": "CPU/Gloo", "rounds": [], "status": "running"}
    for number in range(1, 6):
        junit = OUT / f"repeat-{number}.xml"
        log = OUT / f"repeat-{number}.log"
        command = [sys.executable, "-m", "pytest", "-q", *TESTS, f"--junitxml={junit}"]
        start = time.monotonic()
        timed_out = False
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        row = {"round": number, "command": command, "wall_seconds": time.monotonic()-start,
               "exit_code": process.returncode, "timed_out": timed_out,
               "log": str(log.relative_to(ROOT)), "junit": str(junit.relative_to(ROOT)),
               "source_hashes_unchanged": source_hashes() == before}
        if junit.exists():
            suites = ET.parse(junit).getroot()
            nodes = [suites] if suites.tag == "testsuite" else list(suites.findall("testsuite"))
            counts = {key: sum(int(s.attrib.get(key, 0)) for s in nodes)
                      for key in ("tests", "failures", "errors", "skipped")}
            row["junit_counts"] = counts
            row["passed"] = counts["tests"]-counts["failures"]-counts["errors"]-counts["skipped"]
        results["rounds"].append(row)
        success = (process.returncode == 0 and not timed_out and row.get("passed") == 8
                   and row["source_hashes_unchanged"] and row["junit_counts"]["skipped"] == 0)
        print(json.dumps(row), flush=True)
        if not success:
            results["status"] = "stopped_on_failure_or_source_change"
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        if not success:
            break
    else:
        results["status"] = "passed_five_independent_cpu_gloo_rounds"
    manifest["source_sha256_after"] = source_hashes()
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_hashes_unchanged"] = manifest["source_sha256_after"] == before
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"status": results["status"], "rounds": len(results["rounds"]), "output": str(OUT)}), flush=True)
    return 0 if results["status"] == "passed_five_independent_cpu_gloo_rounds" else 1


if __name__ == "__main__":
    raise SystemExit(main())
