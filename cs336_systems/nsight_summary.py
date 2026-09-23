"""Read-only Nsight analysis with process-aware CUDA launch attribution.

Top-level phases include autograd worker threads. Fine attention ranges remain
thread-local. This assumes serialized steps, not concurrent unrelated jobs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3


PROCESS_RANGES = {"measurement", "forward", "backward", "loss", "optimizer", "gradient_sync_wait"}
ATTENTION_RANGES = {"attention_scores_matmul", "attention_mask", "attention_softmax", "attention_values_matmul"}


def process_identity(global_tid):
    """Nsight's encoded globalPid keeps all but the lower 24 thread-ID bits."""
    return int(global_tid) & ~((1 << 24) - 1)


def summarize(path, output_dir=None):
    """Return summary, optionally writing derived files outside the source.

    Missing GPU evidence is explicit, not a claim of zero GPU work. The source
    database is always opened read-only; no outputs are written by default.
    """
    path = Path(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        schema = {name: [r[1] for r in db.execute('PRAGMA table_info("' + name.replace('"', '""') + '")')]
                  for name in tables}
        def rows(name):
            return [dict(r) for r in db.execute('SELECT * FROM "' + name.replace('"', '""') + '"')] if name in tables else []
        strings = {r["id"]: r["value"] for r in rows("StringIds")}
        kernels = rows("CUPTI_ACTIVITY_KIND_KERNEL")
        runtime = rows("CUPTI_ACTIVITY_KIND_RUNTIME")
        ranges = [r for r in rows("NVTX_EVENTS") if r.get("end") is not None and r.get("start") is not None
                  and r["end"] >= r["start"]]
        diagnostics = rows("DIAGNOSTIC_EVENT")
        processes = rows("PROCESSES")
    result = {"sqlite_source": str(path), "sqlite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "tables": schema, "processes": processes, "status": "ok" if kernels else "unsupported_missing_gpu_kernels",
              "attribution_policy": {"launch_key": "(encoded globalPid, correlationId)",
                  "process_id_from_thread": "globalTid & ~((1 << 24) - 1)",
                  "process_wide_ranges": sorted(PROCESS_RANGES), "thread_local_ranges": sorted(ATTENTION_RANGES),
                  "scope": "CPU launch start within NVTX interval, not GPU timestamp containment",
                  "limitation": "Process-wide phase attribution assumes serialized model steps; concurrent unrelated work is ambiguous.",
                  "matmul_classification": "Kernel name contains gemm/gemv, case-insensitive; disclosed heuristic, not FLOPs."},
              "diagnostics": [{**r, "message": strings.get(r.get("text"), r.get("text"))} for r in diagnostics]}
    def label(r):
        return r.get("text") or strings.get(r.get("textId"), "")
    launches = defaultdict(list)
    for r in runtime:
        if r.get("globalTid") is not None and r.get("correlationId") is not None:
            launches[(process_identity(r["globalTid"]), r["correlationId"])].append(r)
    for group in launches.values():
        group.sort(key=lambda r: r["start"])
    unmatched = ambiguous = 0
    for i, k in enumerate(kernels):
        k["index"] = i
        k["name"] = strings.get(k.get("demangledName"), strings.get(k.get("shortName"), "unknown"))
        k["duration_ms"] = (k["end"] - k["start"]) / 1e6
        k["is_matmul"] = bool(re.search(r"gemm|gemv", k["name"], re.IGNORECASE))
        candidates = launches.get((k.get("globalPid"), k.get("correlationId")), [])
        if not candidates:
            k["launch"] = None
            unmatched += 1
        else:
            ambiguous += int(len(candidates) > 1)
            preceding = [r for r in candidates if r["start"] <= k["start"]]
            k["launch"] = preceding[-1] if preceding else candidates[0]
    def aggregate(items):
        groups = defaultdict(lambda: {"count": 0, "gpu_ms": 0.0})
        total = sum(k["duration_ms"] for k in items)
        for k in items:
            g = groups[k["name"]]
            g["count"] += 1
            g["gpu_ms"] += k["duration_ms"]
            g["is_matmul"] = k["is_matmul"]
        return sorted([{"name": n, **g, "percent_gpu_kernel_time": 100*g["gpu_ms"]/total if total else None}
                       for n, g in groups.items()], key=lambda g: -g["gpu_ms"])
    def metrics(items):
        total = sum(k["duration_ms"] for k in items)
        matmul = sum(k["duration_ms"] for k in items if k["is_matmul"])
        return {"kernel_count": len(items), "gpu_kernel_ms": total, "matmul_gpu_ms": matmul,
                "matmul_fraction": matmul/total if total else None,
                "gpu_span_ms": (max(k["end"] for k in items)-min(k["start"] for k in items))/1e6 if items else None,
                "kernel_summary": aggregate(items)}
    result["kernel_summary"] = aggregate(kernels)
    result["total_gpu_kernel_ms"] = sum(k["duration_ms"] for k in kernels)
    result["kernel_count"] = len(kernels)
    result["unmatched_launch_kernel_count"] = unmatched
    result["ambiguous_launch_kernel_count"] = ambiguous
    result["nvtx_range_labels"] = sorted(set(label(r) for r in ranges))[:400]
    by_range = {}
    selected_by_range = {}
    for name in sorted(PROCESS_RANGES | ATTENTION_RANGES):
        intervals = [r for r in ranges if label(r) == name]
        selected = []
        for k in kernels:
            launch = k["launch"]
            if launch is None:
                continue
            for r in intervals:
                thread_matches = r.get("globalTid") == launch["globalTid"]
                process_matches = r.get("globalTid") is not None and process_identity(r["globalTid"]) == process_identity(launch["globalTid"])
                if (process_matches if name in PROCESS_RANGES else thread_matches) and r["start"] <= launch["start"] < r["end"]:
                    selected.append(k)
                    break
        selected_by_range[name] = selected
        by_range[name] = {"range_count": len(intervals), "cpu_ms": sum((r["end"]-r["start"])/1e6 for r in intervals),
                          "attribution_scope": "process" if name in PROCESS_RANGES else "thread", **metrics(selected)}
    union_indices = {k["index"] for n in ("forward", "loss", "backward") for k in selected_by_range[n]}
    by_range["forward_loss_backward"] = metrics([k for k in kernels if k["index"] in union_indices])
    memberships = defaultdict(list)
    for name in ("forward", "loss", "backward", "optimizer", "gradient_sync_wait"):
        for k in selected_by_range[name]:
            memberships[k["index"]].append(name)
    result["phase_audit"] = {"measurement_kernel_count": len(selected_by_range["measurement"]),
                             "measurement_unassigned_count": sum(not memberships[k["index"]] for k in selected_by_range["measurement"]),
                             "multiply_assigned_phase_kernel_count": sum(len(v)>1 for v in memberships.values())}
    result["ranges"] = by_range
    start = min(k["start"] for k in kernels) if kernels else 0
    timeline = {"origin_ns": start,
                "kernels": [{"start_ms": (k["start"]-start)/1e6, "duration_ms": k["duration_ms"],
                             "name": k["name"], "stream": k.get("streamId"), "device": k.get("deviceId"),
                             "globalPid": k.get("globalPid"), "launch_globalTid": k["launch"]["globalTid"] if k["launch"] else None,
                             "phases": memberships[k["index"]]} for k in kernels],
                "ranges": [{"start_ms": (r["start"]-start)/1e6, "duration_ms": (r["end"]-r["start"])/1e6,
                            "name": label(r), "globalTid": r.get("globalTid")} for r in ranges if label(r) in by_range]}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "nsys_timeline.json").write_text(json.dumps(timeline))
        (output_dir / "nsys_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Optional derived-output directory; source is never modified.")
    args = parser.parse_args()
    report = summarize(args.sqlite, args.output_dir)
    print(json.dumps({"status": report["status"], "kernels": report["kernel_count"],
                      "total_gpu_kernel_ms": report["total_gpu_kernel_ms"], "phase_audit": report["phase_audit"],
                      "ranges": {n: {k: v for k, v in d.items() if k != "kernel_summary"}
                                 for n, d in report["ranges"].items()}}, indent=2))
