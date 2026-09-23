"""Reduce real Nsight SQLite events without interpreting CPU time as GPU time."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys


def summarize(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    schema = {name: [r[1] for r in db.execute(f'PRAGMA table_info("{name}")')] for name in tables}
    result = {"sqlite_source": path.name, "tables": schema}
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    kernels = [dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    runtime = {r["correlationId"]: dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    ranges = [dict(r) for r in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    def label(r):
        return r.get("text") or strings.get(r.get("textId"), "")
    def kernel_name(k):
        return strings.get(k.get("demangledName"), strings.get(k.get("shortName"), "unknown"))
    for k in kernels:
        k["name"] = kernel_name(k)
        k["duration_ms"] = (k["end"] - k["start"]) / 1e6
    def aggregate(items):
        groups = defaultdict(lambda: {"count": 0, "gpu_ms": 0.0})
        for k in items:
            g = groups[k["name"]]
            g["count"] += 1
            g["gpu_ms"] += k["duration_ms"]
        return sorted([{"name": n, **g} for n, g in groups.items()], key=lambda g: -g["gpu_ms"])
    result["kernel_summary"] = aggregate(kernels)
    result["total_gpu_kernel_ms"] = sum(k["duration_ms"] for k in kernels)
    result["nvtx_range_labels"] = sorted(set(label(r) for r in ranges))[:400]
    by_range = {}
    for name in ("measurement", "forward", "backward", "loss", "optimizer", "gradient_sync_wait",
                 "attention_scores_matmul", "attention_mask", "attention_softmax", "attention_values_matmul"):
        intervals = [r for r in ranges if label(r) == name]
        selected = []
        for k in kernels:
            launch = runtime.get(k["correlationId"])
            if launch is not None and any(r["start"] <= launch["start"] <= r["end"]
                                         and r["globalTid"] == launch["globalTid"] for r in intervals):
                selected.append(k)
        by_range[name] = {"range_count": len(intervals), "cpu_ms": sum((r["end"]-r["start"])/1e6 for r in intervals),
                          "gpu_kernel_ms": sum(k["duration_ms"] for k in selected),
                          "kernel_summary": aggregate(selected)}
    result["ranges"] = by_range
    # Compact real-event timeline for offline viewing and supplementary figures.
    start = min(k["start"] for k in kernels) if kernels else 0
    timeline = {"origin_ns": start,
                "kernels": [{"start_ms": (k["start"]-start)/1e6, "duration_ms": k["duration_ms"],
                             "name": k["name"], "stream": k["streamId"], "device": k["deviceId"]} for k in kernels],
                "ranges": [{"start_ms": (r["start"]-start)/1e6, "duration_ms": (r["end"]-r["start"])/1e6,
                            "name": label(r)} for r in ranges if label(r) in by_range]}
    (path.parent / "nsys_timeline.json").write_text(json.dumps(timeline))
    (path.parent / "nsys_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"kernels": len(kernels), "total_gpu_kernel_ms": result["total_gpu_kernel_ms"],
                      "ranges": {n: {k: v for k, v in d.items() if k != "kernel_summary"} for n, d in by_range.items()}}, indent=2))


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
