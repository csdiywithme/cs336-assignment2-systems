"""Build a local-only evidence book. No network, cloud launch, or raw-data edits.

Python 3.11+, standard library only. Generated artifacts go to output/data and
output/tables. A missing result is unknown, never an inferred failure.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA, TABLES = ROOT / "output/data", ROOT / "output/tables"
SIZES = ("small", "medium", "large", "xl", "10B")
MODES = ("forward", "backward", "train")
LABELS = {"forward": "F", "backward": "F+CE+B", "train": "F+CE+B+AdamW"}
RATES = {"B200_per_gpu_second": .001736, "cpu_per_core_second": .0000131,
         "ram_per_GiB_second": .00000222, "cpu_cores": 8, "ram_GiB": 64}


def read(p):
    return json.loads(p.read_text())


def rel(p):
    return str(p.relative_to(ROOT))


def dump(name, obj):
    (DATA / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def stats(values):
    return {"samples_ms": values, "mean_ms": statistics.mean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.,
            "median_ms": statistics.median(values)}


def junit(p):
    root = ET.parse(p).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    obj = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    obj["passed"] = obj["tests"] - obj["failures"] - obj["errors"] - obj["skipped"]
    obj["seconds"] = sum(float(s.get("time", 0)) for s in suites)
    return obj


def load():
    attempts, runs = {}, {}
    for p in sorted((ROOT / "runs/dispatch").glob("*/dispatch.json")):
        d = read(p)
        q = p.with_name("request.json")
        request = read(q) if q.exists() else {}
        runs[d["run_id"]] = {"run_id": d["run_id"], "run_code": f"R{len(runs)+1:02}",
            "dispatch": d, "dispatch_source": rel(p), "request_source": rel(q),
            "requested_cases": request.get("cases", []), "provenance": request.get("provenance", {}),
            "summary_sources": [], "local_files": [], "remote_inventory": []}

    def add(rid, case_dir, data, p, pointer="", result_only=False):
        name = case_dir.split("-", 1)[1]
        cfg = next((c for c in runs[rid]["requested_cases"] if c["name"] == name), {"name": name})
        result = data if result_only else data.get("result", {})
        a = attempts.setdefault((rid, case_dir), {"config": deepcopy(cfg), "measurements": {},
                "run_id": rid, "case_dir": case_dir, "status": "unknown", "outer_status": None, "sources": []})
        ref = {"path": rel(p), "pointer": pointer, "kind": "result" if result_only else "wrapper"}
        if ref not in a["sources"]:
            a["sources"].append(ref)
        if not result_only:
            for source, target in (("status", "outer_status"), ("returncode", "returncode"),
                    ("seconds", "outer_wall_seconds"), ("trace_created", "trace_created"), ("command", "command")):
                if source in data:
                    a[target] = data[source]
            if not a.get("result_status"):
                a["status"] = data.get("status", "unknown")
        if result:
            a.update(deepcopy(result))
            a["result_status"] = result.get("status")
            a.update(source=rel(p), source_pointer=pointer + ("" if result_only else "/result"))
        a.setdefault("source", rel(p))
        a.setdefault("source_pointer", pointer)

    def summary(rid, data, p, pointer=""):
        runs[rid]["summary"] = data
        runs[rid]["summary_sources"].append({"path": rel(p), "pointer": pointer})
        for i, case in enumerate(data.get("cases", [])):
            add(rid, f"{i:03}-{case['name']}", case, p, pointer + f"/cases/{i}")

    # Status files contain actual returned FunctionCall results; they are valid
    # numeric evidence even when the case-level files have not been downloaded.
    for p in sorted((ROOT / "runs/status").glob("*.json")):
        obj = read(p).get("result")
        if isinstance(obj, dict) and "cases" in obj:
            summary(obj["run_id"], obj, p, "/result")
    for p in sorted((ROOT / "runs/cloud").glob("*/*/summary.json")):
        summary(p.relative_to(ROOT / "runs/cloud").parts[0], read(p), p)
    for p in sorted((ROOT / "runs/cloud").glob("*/*/progress.json")):
        rid = p.relative_to(ROOT / "runs/cloud").parts[0]
        for i, case in enumerate(read(p)):
            add(rid, f"{i:03}-{case['name']}", case, p, f"/{i}")
    for filename in ("status.json", "result.json"):
        for p in sorted((ROOT / "runs/cloud").glob(f"*/*/*/{filename}")):
            rid, _, case_dir = p.relative_to(ROOT / "runs/cloud").parts[:3]
            try:
                obj = read(p)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                runs[rid].setdefault("unreadable_local_artifacts", []).append({"source": rel(p), "error": str(error)})
                continue
            add(rid, case_dir, obj, p, result_only=filename == "result.json")

    for rid, run in runs.items():
        cloud = ROOT / "runs/cloud" / rid
        run["local_files"] = [rel(p) for p in sorted(cloud.rglob("*")) if p.is_file()]
        for p in sorted(cloud.glob("*/artifact_inventory.json")):
            run["remote_inventory"] = read(p)
        local_names = {str(Path(p).relative_to(Path("runs/cloud") / rid)).split("/", 1)[1] for p in run["local_files"]}
        inventory = [p for p in run["remote_inventory"] if not any("cache" in part for part in Path(p["path"]).parts)]
        run["artifact_availability"] = {
            "summary": bool(run["summary_sources"]), "cloud_snapshot": bool(run["local_files"]),
            "environment": any(p.endswith("/environment.json") for p in run["local_files"]),
            "source_snapshot": any("/source/" in p for p in run["local_files"]),
            "remote_manifest_files_not_local": [p for p in inventory if p["path"] not in local_names],
            "full_artifact_set_verified": bool(inventory) and all(p["path"] in local_names for p in inventory)}
        for (arid, case_dir), a in attempts.items():
            if arid != rid:
                continue
            files = [p for p in run["local_files"] if f"/{case_dir}/" in p]
            a["artifact_availability"] = {"local_case_files": files, "status_snapshot_only": not files}
            for key, suffix in (("result_json", "/result.json"), ("stdout", "/stdout.log"), ("junit", "/junit.xml"),
                    ("memory_snapshot", ".pickle"), ("nsight_sqlite", ".sqlite"), ("nsight_report", ".nsys-rep")):
                a["artifact_availability"][key] = any(p.endswith(suffix) for p in files)
            for file in files:
                p = ROOT / file
                if p.name == "junit.xml":
                    a["junit"] = junit(p)
                if p.name == "partial.json":
                    a["partial_measurements"] = read(p)
            a.update(run_code=run["run_code"], gpu_count=run["dispatch"].get("gpu_count"))
            derive(a)
    ordered = [a for _, a in sorted(attempts.items())]
    for i, a in enumerate(ordered, 1):
        a["source_index"] = f"E{i:03}"
    for p in sorted((ROOT / "output/profiling").glob("*/nsys_summary.json")):
        report = read(p)
        source = Path(report["sqlite_source"])
        if source.is_absolute():
            source = source.relative_to(ROOT)
        parts = source.parts
        if len(parts) < 6 or parts[:2] != ("runs", "cloud"):
            continue
        a = attempts.get((parts[2], parts[4]))
        if a is not None:
            a["profiling_evidence"] = {"analysis_source": rel(p), "sqlite_source": str(source),
                "sqlite_sha256": report.get("sqlite_sha256"), "status": report["status"],
                "kernel_count": report.get("kernel_count"), "total_gpu_kernel_ms": report.get("total_gpu_kernel_ms"),
                "unmatched_launch_kernel_count": report.get("unmatched_launch_kernel_count"),
                "ambiguous_launch_kernel_count": report.get("ambiguous_launch_kernel_count"),
                "phase_audit": report.get("phase_audit"), "attribution_policy": report.get("attribution_policy")}
    return ordered, list(runs.values())


def derive(a):
    c, m = a["config"], a.get("measurements", {})
    d = a.setdefault("derived_metrics", {})
    if c.get("kind") == "model":
        d["timing_semantics"] = LABELS.get(c["mode"])
        d["forward_records_gradients"] = True
        d["default_batch_seq_vocab"] = [c.get("batch", 4), c.get("seq", 512), c.get("vocab", 10000)]
        d["memory_semantics"] = "Extra recorded step after measured steps; peak reset before this extra step."
    if c.get("kind") == "distributed" and m.get("ranks"):
        for key in ("step", "exposed_sync"):
            d[f"rank_max_{key}"] = stats([max(r[key]["samples_ms"][i] for r in m["ranks"])
                for i in range(len(m["ranks"][0][key]["samples_ms"]))])
        d["communication_semantics"] = "Tail wait after synchronized backward, NOT total communication time or overlap ratio."
        d["memory_rank_max"] = {"initial_allocated_bytes": max(r["memory_after_initialization"]["allocated_bytes"] for r in m["ranks"]),
            "before_optimizer_allocated_bytes": max(s["before_optimizer"]["allocated_bytes"] for r in m["ranks"] for s in r["memory_stages"]),
            "after_optimizer_allocated_bytes": max(s["after_optimizer"]["allocated_bytes"] for r in m["ranks"] for s in r["memory_stages"]),
            "peak_allocated_bytes": max(s["after_optimizer"]["peak_allocated_bytes"] for r in m["ranks"] for s in r["memory_stages"])}
    if c.get("kind") == "leaderboard" and m.get("ranks"):
        d["cold_start_total_seconds_rank_max"] = max(r["cold_start_total_seconds"] for r in m["ranks"])
        d["measured_peak_allocated_bytes_rank_max"] = max(s["peak_allocated_bytes"] for r in m["ranks"] for s in r["measurements"] if not s["warmup"])
    if c.get("kind") == "allreduce" and m.get("ranks"):
        d["recomputed_rank_max"] = stats([max(r["samples_ms"][i] for r in m["ranks"])
                                          for i in range(c["steps"])])
        d["rank_max_matches_raw"] = all(math.isclose(x, y, abs_tol=1e-7, rel_tol=1e-7)
            for x, y in zip(d["recomputed_rank_max"]["samples_ms"], m["rank_max"]["samples_ms"]))
        d["recomputed_bus_bandwidth_GBps"] = 2 * (c["world"] - 1) / c["world"] * c["bytes"] / (d["recomputed_rank_max"]["mean_ms"] * 1e6)
        d["bus_bandwidth_matches_raw"] = math.isclose(d["recomputed_bus_bandwidth_GBps"], m["rank_max"]["bus_bandwidth_GBps"], abs_tol=1e-7, rel_tol=1e-7)
    checks = []
    def visit(value, pointer):
        if isinstance(value, dict):
            if "samples_ms" in value:
                xs = value["samples_ms"]
                fresh = stats(xs) if xs else {}
                errors = [k for k in ("mean_ms", "std_ms", "median_ms") if k in value and not
                    math.isclose(value[k], fresh.get(k, math.nan), rel_tol=1e-7, abs_tol=1e-7)]
                checks.append({"path": pointer, "observed_n": len(xs), "expected_n": c.get("steps"),
                    "count_ok": "steps" not in c or len(xs) == c["steps"], "statistics_ok": not errors, "statistic_mismatches": errors})
            for k, v in value.items():
                visit(v, pointer + "/" + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                visit(v, pointer + f"/{i}")
    visit(m, "/measurements")
    visit(d, "/derived_metrics")
    finite = True
    if c.get("triton_bench") and a["status"] == "ok":
        finite = all(math.isfinite(m.get(k, {}).get("mean_ms", math.nan)) and m[k]["mean_ms"] > 0 for k in ("forward", "backward", "end_to_end"))
    a["sample_validation"] = {"checks": checks, "all_three_do_bench_means_finite_positive": finite,
        "passed": finite and all(x["count_ok"] and x["statistics_ok"] for x in checks) and d.get("rank_max_matches_raw", True) and d.get("bus_bandwidth_matches_raw", True),
        "timing_method_note": "do_bench warmup=100ms, rep=300ms; internal n/raw samples/SD unrecorded; config steps=10 and warmup=5 are NOT used in this branch." if c.get("triton_bench") else None}


def ledger(runs):
    entries, missing = [], []
    for r in runs:
        dispatch = r["dispatch"]
        row = {"run_id": r["run_id"], "run_code": r["run_code"], "gpu_count": dispatch["gpu_count"],
               "job_timeout_seconds": dispatch["timeout_seconds"], "dispatch_source": r["dispatch_source"]}
        if not r.get("summary"):
            missing.append({**row, "reason": "Local completed-job summary missing; cost unknown and excluded."})
            continue
        summary = r["summary"]
        rate = dispatch["gpu_count"] * RATES["B200_per_gpu_second"] + 8 * RATES["cpu_per_core_second"] + 64 * RATES["ram_per_GiB_second"]
        entries.append({**row, "runtime_seconds": summary["seconds"], "raw_estimate_usd": summary.get("estimated_compute_usd"),
            "corrected_runtime_estimate_usd": summary["seconds"] * rate, "rate_per_second": rate,
            "source": r["summary_sources"][-1]["path"], "sources": r["summary_sources"], "case_count": len(summary.get("cases", []))})
    return {"currency": "USD", "price_source": "https://modal.com/pricing", "rates": RATES,
        "note": "INCOMPLETE runtime-only subtotal, NOT invoice or total spend. Excludes missing-summary jobs, startup/imports/image build/storage/CPU artifact jobs. Early multi-GPU raw estimates corrected using exact dispatch GPU count. Raw files unchanged.",
        "runs": entries, "missing_summary_runs": missing, "is_complete_total": False, "is_invoice": False,
        "total_corrected_runtime_estimate_usd": sum(r["corrected_runtime_estimate_usd"] for r in entries)}


def matrices(attempts, local_tests=()):
    byname = {a["config"]["name"]: a for a in attempts}
    names = {
        "models": [f"{s}-{d}-{m}-w{w}" for s in SIZES for d, w in (("fp32", 5), ("bf16", 5), ("fp32", 0), ("fp32", 1), ("fp32", 2)) for m in MODES],
        "models_compile_including_missing_train": [f"{s}-compiled-{m}" for s in SIZES for m in MODES],
        "attention": [f"attention-{i}-s{s}-d{d}" for i in ("eager", "compiled") for s in (256, 1024, 4096, 8192, 16384) for d in (16, 32, 64, 128)],
        "flash": [f"flash-{i}-{t}-s{2**s}-d{d}" for i in ("eager", "triton", "triton-full") for t in ("fp32", "bf16") for s in range(7, 17) for d in (16, 32, 64, 128)],
        "communication": [f"{b}-{w}ranks-{mb}MB" for w in (2, 4, 6) for b in ("gloo", "nccl") for mb in (1, 10, 100, 1000)],
        "distributed_xl": [f"xl-{s}-{o}" for s, o in (("naive", "adamw"), ("flat", "adamw"), ("overlap", "adamw"), ("overlap", "sharded"), ("fsdp", "adamw"))],
        "memory": [f"memory-xl-s{s}-{d}-{m}" for s in (128, 2048) for d in ("fp32", "bf16") for m in ("forward", "train")],
        "checkpoint": [f"checkpoint-{s}" for s in (1, 2, 4, 8, 16, 32, 64)],
        "distributed_repeat_recommended": [f"distributed-repeat-{i}" for i in range(5)]}
    result = {key: [{"configuration": {"name": n}, "status": byname[n]["status"] if n in byname else "missing_local_evidence",
                  "source_index": byname[n]["source_index"] if n in byname else None,
                  "run_id": byname[n]["run_id"] if n in byname else None} for n in ns] for key, ns in names.items()}
    groups = defaultdict(list)
    for t in local_tests:
        parent = Path(t["source"]).parent
        if parent.name.startswith("final-gloo-stability"):
            groups[str(parent)].append(t)
    def group_started(directory):
        manifest = ROOT / directory / "manifest.json"
        return read(manifest).get("started_utc", "") if manifest.exists() else ""
    # Different attempts stay separate; five rounds must belong to one frozen
    # source manifest, never a mixture of earlier failed and later good runs.
    newest = max(groups, key=group_started) if groups else None
    stability = sorted(groups[newest], key=lambda t: t["source"]) if newest else []
    if stability:
        rows = []
        for i in range(5):
            t = stability[i] if i < len(stability) else None
            rows.append({"configuration": {"name": f"local-cpu-gloo-repeat-{i}", "device": "cpu", "backend": "gloo"},
                "status": "missing_local_evidence" if not t else ("ok" if t["passed"] == 8 and not (t["failures"] + t["errors"] + t["skipped"]) else "failed"),
                "source_index": None, "run_id": None, "local_test_source": t["source"] if t else None,
                "note": "Original course distributed suites repeated locally; NOT CUDA/NCCL validation or another GPU job."})
        result["distributed_repeat_recommended"] = rows
    return result


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def metric(m):
    return "—" if not m or "mean_ms" not in m else fmt(m["mean_ms"]) + (f" ± {fmt(m['std_ms'])}" if "std_ms" in m else "")


def gib(x):
    return fmt(x / 2**30) if x is not None else "—"


def make_tables(attempts, runs, costs, coverage, tests):
    out = ["# 实验数据全表（本地已有证据）", "",
        "时间单位为 ms；± 为样本标准差（ddof=1），不是置信区间。GiB=2³⁰ bytes；通信 MB=10⁶ bytes。E 编号指向 evidence_index.json，R 编号见文末。"
        "n=1 时脚本保存的 SD=0 仅是占位，不能推断稳定性。结果 ok 不等于 Nsight GPU kernel 已捕获，也不表示所有附件已在本地。本次整理不启动 GPU/云端计算。", ""]
    def section(s, note="", level=2):
        out.extend(["#" * level + " " + s, ""])
        if note:
            out.extend([note, ""])
    def table(headers, rows):
        clean = lambda x: str(x).replace("|", "/").replace("\n", " ")
        out.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        out.extend("| " + " | ".join(clean(v) for v in row) + " |" for row in rows)
        out.append("")
    byname = {a["config"]["name"]: a for a in attempts}
    get = lambda name: byname.get(name)
    ref = lambda a: f"{a['source_index']}/{a['run_code']}" if a else "无本地证据"
    state = lambda a: a["status"] if a else "缺失"
    measure = lambda a: a.get("measurements", {}) if a else {}
    section("0. 覆盖概况")
    table(["矩阵", "计划格子", "ok", "OOM", "失败", "缺失"], [(k, v["expected"], v["statuses"].get("ok", 0), v["statuses"].get("oom", 0), v["statuses"].get("failed", 0), v["statuses"].get("missing_local_evidence", 0)) for k, v in coverage["matrices"].items()])
    if any(Path(t["source"]).parent.name.startswith("final-gloo-stability") for t in tests):
        out.extend(["建议五轮稳定性采用 final-gloo-stability 下本地 CPU/Gloo JUnit，未新增 GPU 实验，也不等于 NCCL 数值验证。历史云端单轮结果仍保留。", ""])
    section("1. 五模型基准、混合精度与预热", "默认 B=4、S=512、词表 10000。F 保留 autograd；F+CE+B 是前向/损失/反向合计，不能叫纯 backward。完整训练再含 staff AdamW。BF16 为 autocast，参数仍 FP32。成功配置各保存 10 次原始计时。")
    for size in SIZES:
        section(size, level=3)
        rows = []
        for dtype, warm in (("fp32", 5), ("bf16", 5), ("fp32", 0), ("fp32", 1), ("fp32", 2)):
            for mode in MODES:
                a = get(f"{size}-{dtype}-{mode}-w{warm}"); m = measure(a)
                rows.append((dtype, warm, LABELS[mode], metric(m), len(m.get("samples_ms", [])) or "—", state(a), ref(a)))
        table(["精度", "预热", "计时范围", "均值 ± SD", "实存 n", "状态", "证据"], rows)
    section("2. torch.compile 模型对照", "FP32、预热 5、每个成功配置 10 次计时。compile 作用于 model。此前计划遗漏五种完整 train 配置，现显式列出，不能用 F+CE+B 代替完整训练。")
    rows = []
    for size in SIZES:
        for mode in MODES:
            a = get(f"{size}-compiled-{mode}"); m = measure(a); base = measure(get(f"{size}-fp32-{mode}-w5"))
            ratio = base["mean_ms"] / m["mean_ms"] if m.get("mean_ms") and base.get("mean_ms") else None
            rows.append((size, LABELS[mode], metric(m), fmt(ratio), len(m.get("samples_ms", [])) or "—", state(a), ref(a)))
    table(["模型", "范围", "compiled 均值 ± SD", "eager/compiled", "n", "状态", "证据"], rows)
    section("3. CUDA event 分阶段计时", "只列原始记录实际保存的事件时间；该口径不同于同步 wall-clock。小/中模型首次 eager 矩阵未加入埋点，不能倒算纯 backward 或 AdamW。")
    rows = []
    for a in attempts:
        c, m = a["config"], measure(a)
        if c.get("kind") != "model" or not m.get("cuda_event_stages") or c.get("nsys") or c.get("memory_history") or c.get("checkpoint_segments") or c["name"].startswith(("fit-", "profile-")):
            continue
        s = m["cuda_event_stages"]
        desc = f"{c['size']}/{c.get('dtype')}/w{c.get('warmup')}/{'compile' if c.get('compile') else 'eager'}/{LABELS[c['mode']]}"
        rows.append((desc, metric(s.get("forward")), metric(s.get("loss")), metric(s.get("backward")), metric(s.get("optimizer")), ref(a)))
    table(["配置", "F", "CE", "B", "AdamW", "证据"], rows)
    section("4. 精度累加与 autocast")
    a = get("precision"); m = measure(a)
    if m:
        table(["累加器", "增量", "结果", "证据"], [(v.get("accumulator", "float32"), v.get("increment", v.get("variant")), repr(v["value"]), ref(a)) for v in m["accumulations"]])
        ac = m["autocast"]
        table(["张量/操作", "实际 dtype", "证据"], [(k, ac[k], ref(a)) for k in ("fc1", "layernorm", "logits", "loss")] + [(f"参数 {k}", v, ref(a)) for k, v in ac["parameters"].items()] + [(f"梯度 {k}", v, ref(a)) for k, v in ac["gradients"].items()])
    section("5. 普通与 compiled attention：40 格", "B=8、非 causal、FP32；F/B/F+B 各预热 5，再计时 100 次。B 是固定前向图的 backward-only。B 前显存是 allocated，不是整步 peak。")
    rows = []
    for impl in ("eager", "compiled"):
        for seq in (256, 1024, 4096, 8192, 16384):
            for dim in (16, 32, 64, 128):
                a = get(f"attention-{impl}-s{seq}-d{dim}"); m = measure(a)
                rows.append((f"{impl}/{seq}/{dim}", metric(m.get("forward")), metric(m.get("backward")), metric(m.get("end_to_end")), gib(m.get("memory_before_backward", {}).get("allocated_bytes")), state(a), ref(a)))
    table(["实现/S/D", "F", "B", "F+B", "B 前 GiB", "状态", "证据"], rows)
    section("6. FlashAttention：240 格", "B=1、causal。eager=普通 PyTorch；triton=手写前向+compiled PyTorch 反向；triton-full=手写前向和反向。调用 do_bench(warmup=100ms,rep=300ms,return_mode=mean)，内部 n/原始样本/SD 未保存，配置 steps=10 未用于此分支。下列为聚合均值；不是 10 个独立样本。")
    for impl in ("eager", "triton", "triton-full"):
        for dtype in ("fp32", "bf16"):
            section(f"{impl} / {dtype}", level=3); rows = []
            for power in range(7, 17):
                for dim in (16, 32, 64, 128):
                    a = get(f"flash-{impl}-{dtype}-s{2**power}-d{dim}"); m = measure(a)
                    if a and a["status"] == "oom":
                        m = a.get("partial_measurements", m)
                    rows.append((2**power, dim, metric(m.get("forward")), metric(m.get("backward")), metric(m.get("end_to_end")), state(a), ref(a)))
            table(["S", "D", "F", "B", "F+B", "状态", "证据"], rows)
    section("7. XL 显存快照实验", "预热 2、计时 2，再额外执行一个记录步骤；峰值是该额外步骤 reset 后的 peak allocated，reserved 是缓存保留量。OOM 不填伪造峰值；远端生成不等于本地持有快照。")
    rows = []
    for seq in (128, 2048):
        for dtype in ("fp32", "bf16"):
            for mode in ("forward", "train"):
                a = get(f"memory-xl-s{seq}-{dtype}-{mode}"); m = measure(a); mem = m.get("memory", {})
                rows.append((f"{seq}/{dtype}", LABELS[mode], metric(m), gib(mem.get("peak_allocated_bytes")), gib(mem.get("peak_reserved_bytes")), state(a), "有" if a and a["artifact_availability"]["memory_snapshot"] else "无", ref(a)))
    table(["S/精度", "范围", "耗时", "峰值 GiB", "保留峰值 GiB", "状态", "本地快照", "证据"], rows)
    section("8. Activation checkpoint", "XL、B=4、S=2048、FP32、32 层；F+CE+B 不含 AdamW，预热 2、计时 3。64 段表示每层 attention/FFN 分开重算。")
    rows = []
    for seg in (1, 2, 4, 8, 16, 32, 64):
        a = get(f"checkpoint-{seg}"); m = measure(a)
        rows.append((seg, "半层" if seg == 64 else f"{32//seg} 层", metric(m), gib(m.get("memory", {}).get("peak_allocated_bytes")), state(a), ref(a)))
    table(["段数", "每段", "耗时", "峰值 GiB", "状态", "证据"], rows)
    section("9. 单 block 保存张量归因", "XL block、B=4、S=2048、D=2560；按 storage 去重并排除参数/缓冲区。这是 autograd 保存存储，不是 kernel 时间。")
    candidates = [a for a in attempts if a["config"].get("kind") == "saved_block" and a["status"] == "ok" and not a["config"].get("nsys")]
    if candidates:
        a = candidates[-1]; m = measure(a); groups = defaultdict(int)
        for r in m.get("records", []):
            if not r.get("duplicate") and not r.get("parameter_or_buffer"):
                groups[(r.get("module"), r.get("producer"))] += r["storage_bytes"]
        total = sum(groups.values())
        out.extend([f"证据 {ref(a)}；全部 records 留存于 JSON。", ""])
        table(["模块", "保存来源", "bytes", "GiB", "占比"], [(module, producer, size, gib(size), f"{size/total*100:.2f}%") for (module, producer), size in sorted(groups.items(), key=lambda p: -p[1])])
        table(["原始统计字段", "值"], [(k, v) for k, v in m.items() if isinstance(v, (int, float, str))])
    section("10. All-reduce：24 格", "每配置预热 5、计时 20；每迭代先取全部 rank 最大值，再求均值/SD。barrier 和 tensor.fill 不计时。Gloo 用 CPU 张量；NCCL 用 GPU 张量。总线带宽=2(N-1)/N×数据量/耗时，不是实测 NVLink 峰值。")
    rows = []
    for world in (2, 4, 6):
        for backend in ("gloo", "nccl"):
            for mb in (1, 10, 100, 1000):
                a = get(f"{backend}-{world}ranks-{mb}MB"); m = measure(a).get("rank_max", {})
                rows.append((world, backend, mb, metric(m), fmt(m.get("bus_bandwidth_GBps")), len(m.get("samples_ms", [])) or "—", ref(a)))
    table(["rank", "后端", "MB", "耗时", "总线 GB/s", "n", "证据"], rows)
    section("11. 双 B200 XL 分布式", "全局 B=4（每 rank 2）、S=512、FP32，预热 5、计时 10。完整步含 F/CE/B/同步/AdamW（以及 owner broadcast）。每迭代取最大 rank。暴露尾部等待不是总通信耗时，也不能直接推算重叠率。")
    rows, memrows = [], []
    for strategy, opt in (("naive", "adamw"), ("flat", "adamw"), ("overlap", "adamw"), ("overlap", "sharded"), ("fsdp", "adamw")):
        a = get(f"xl-{strategy}-{opt}"); d = a.get("derived_metrics", {}) if a else {}; mem = d.get("memory_rank_max", {})
        rows.append((strategy, opt, metric(d.get("rank_max_step")), metric(d.get("rank_max_exposed_sync")), state(a), ref(a)))
        memrows.append((strategy, opt, *[gib(mem.get(k)) for k in ("initial_allocated_bytes", "before_optimizer_allocated_bytes", "after_optimizer_allocated_bytes", "peak_allocated_bytes")], ref(a)))
    table(["策略", "优化器", "完整步", "尾部等待", "状态", "证据"], rows)
    table(["策略", "优化器", "初始化 GiB", "更新前 GiB", "更新后 GiB", "步峰值 GiB", "证据"], memrows)
    section("12. Nsight 与上下文容量探测", "Nsight wall-clock 有采集开销，不替代普通基准。trace_created 只说明 SQLite 生成，不保证有 GPU kernel。早期 HES 采集缺 kernel；取回后已确认最后 small/S256 cuda-sw 探针有效。只认这一个 case，不把其他尚无有效 kernel 证据的 profile 一并算完成。")
    rows = []
    for a in attempts:
        c = a["config"]
        if c.get("nsys") or c["name"].startswith(("fit-", "profile-")):
            m = measure(a) if c.get("kind") == "model" else a["derived_metrics"].get("rank_max_step", {})
            profile = a.get("profiling_evidence", {})
            rows.append((c["name"], metric(m), state(a), a.get("trace_created", "—"), "有" if a["artifact_availability"]["nsight_sqlite"] else "无", profile.get("kernel_count", "未验证"), ref(a)))
    table(["case", "wall-clock", "结果", "trace_created", "本地 SQLite", "有效 kernels", "证据"], rows)
    for a in attempts:
        evidence = a.get("profiling_evidence")
        if not evidence or evidence["status"] != "ok":
            continue
        profile = read(ROOT / evidence["analysis_source"])
        section(f"有效 kernel 统计：{a['config']['name']}（{ref(a)}）", "按 (globalPid, correlationId) 关联 CUDA launch；顶层 phase 跨 autograd worker 线程，attention 细分 range 只关联同线程。串行训练步假设下，无未匹配/歧义 launch，measurement 内 phase 无漏分/重复。GPU 时间是 kernel 时长之和，不是 wall-clock，也不是跨 range 首尾 span。矩阵乘法分类依据名称含 gemm/gemv，是启发式而非 FLOPs 测量。", level=3)
        table(["统计", "实测值"], [("kernel 总数", profile["kernel_count"]), ("kernel 时间总和 ms", fmt(profile["total_gpu_kernel_ms"], 6)),
            ("未匹配 launch", profile["unmatched_launch_kernel_count"]), ("歧义 launch", profile["ambiguous_launch_kernel_count"]),
            *profile["phase_audit"].items()])
        rows = []
        for label, r in profile["ranges"].items():
            rows.append((label, r.get("kernel_count"), fmt(r.get("cpu_ms")), fmt(r.get("gpu_kernel_ms"), 6),
                fmt(r.get("matmul_gpu_ms"), 6), fmt(100*r["matmul_fraction"]) + "%" if r.get("matmul_fraction") is not None else "—"))
        table(["range", "kernels", "CPU range ms", "GPU kernel ms", "matmul ms", "matmul 比例"], rows)
        table(["耗时前 10 kernel（完整名称见 JSON）", "调用数", "GPU ms", "占总 kernel 时间"],
            [(k["name"][:150] + ("…" if len(k["name"]) > 150 else ""), k["count"], fmt(k["gpu_ms"], 6), fmt(k["percent_gpu_kernel_time"]) + "%") for k in profile["kernel_summary"][:10]])
        out.extend([f"分析来源：{evidence['analysis_source']}。此记录不能替代其他 5 个上下文/模型配置、双卡通信重叠、FSDP 预取或逐配置内存截图。", ""])
    section("13. 正确性与历史测试", "通过仅覆盖当时源码与测试容差，最终 GPU 源码重验另列待跑。原课程 distributed tests 硬编码 Gloo；在 GPU 机器选择 CUDA tensor 也不会自动换成 NCCL。历史失败保留；测试次数不与不同配置个数混淆。")
    rows = []
    for a in attempts:
        c = a["config"]
        if c.get("pytest"):
            j = a.get("junit", {})
            rows.append((c["name"], *[j.get(k, "—") for k in ("passed", "skipped", "failures", "errors")], state(a), ref(a)))
        elif c.get("kind") == "validate_attention":
            rows.append((c["name"], f"{len(measure(a).get('checks', []))} 配置检查", "—", "—", "—", state(a), ref(a)))
    table(["GPU 验证", "通过", "跳过", "失败", "异常", "结果", "证据"], rows)
    table(["本地 XML（相对 runs/local）", "通过", "跳过", "失败", "异常", "秒"], [(str(Path(t["source"]).relative_to("runs/local")), t.get("passed", "—"), t.get("skipped", "—"), t.get("failures", "—"), t.get("errors", "—"), fmt(t.get("seconds"))) for t in tests])
    for a in attempts:
        if a["config"].get("kind") == "validate_attention" and a["status"] == "ok":
            section(f"扩展 attention：32 配置 / 128 张量比较（{ref(a)}）", "2 精度 × 4 形状 × 2 causal × 2 实现 = 32 配置；每配置比较 O、dQ、dK、dV。FP32 atol=1e-3；BF16 atol=0.03；两者 rtol=0.03。不能把此扩展测试的宽容差冒充课程全部官方容差。", level=3)
            table(["实现", "dtype", "N/D/causal", "O 最大误差", "dQ 最大误差", "dK 最大误差", "dV 最大误差"],
                  [(v["implementation"].replace("FlashAttention", ""), v["dtype"].replace("torch.", ""), f"{v['n']}/{v['d']}/{v['causal']}",
                    *[f"{e:.8g}" for e in v["max_abs_errors_output_dq_dk_dv"]]) for v in measure(a)["checks"]])
    for a in attempts:
        if a["config"].get("kind") == "validate_optimized":
            section(f"{a['config']['name']}（{ref(a)}，{state(a)}）", level=3); m = measure(a)
            values = [(k, f"{v:.10g}") for k, v in m.items() if isinstance(v, (int, float))]
            values += [(f"gradient/{k}", f"{v:.10g}") for k, v in m.get("model_gradient_errors", {}).items()]
            if values:
                table(["检查项（最大绝对误差，除非另标）", "值"], values)
            else:
                out.extend(["此历史尝试在断言处失败，没有 measurements 汇总；异常完整保留在原始 JSON，末行见历史异常表。不能将未生成的数据补成 0。", ""])
    section("14. 8B / 32K 完整训练步", "34 层、D=4096、FFN=11008、32 heads、词表 151936、S=32768、全局 B=2；双 B200，BF16 compute+FP32 参数。每步包含 F/分块 CE/B/梯度 all-reduce/AdamW/owner broadcast。每次取两 rank 最大同步 wall-clock；新缓存路径验证冷启动。均值是同一 run 稳态样本，不是多次独立冷启动分布。")
    leaders = [a for a in attempts if a["config"].get("kind") == "leaderboard" and a["status"] == "ok"]
    table(["版本", "预热", "n", "完整步 ms", "rank 冷启动总秒", "测量步峰值 GiB", "证据"], [("tuned" if a["config"].get("tuned") else "first", a["config"]["warmup"], len(a["measurements"]["rank_max_step"]["samples_ms"]), metric(a["measurements"]["rank_max_step"]), fmt(a["derived_metrics"].get("cold_start_total_seconds_rank_max")), gib(a["derived_metrics"].get("measured_peak_allocated_bytes_rank_max")), ref(a)) for a in leaders])
    table(["版本", "实际计时样本 ms", "证据"], [("tuned" if a["config"].get("tuned") else "first", ", ".join(fmt(v, 6) for v in a["measurements"]["rank_max_step"]["samples_ms"]), ref(a)) for a in leaders])
    section("15. 样本与附件审计", "逐个带 samples_ms 的对象重算均值、中位数、样本标准差并核对 n。do_bench 仅有聚合均值，不伪造原始样本或 SD。")
    table(["审计项", "结果"], [("原始统计对象", coverage["sample_audit"]["statistic_objects"]), ("不一致", coverage["sample_audit"]["issues"]), ("do_bench 配置", coverage["sample_audit"]["do_bench_cases"]), *coverage["artifacts"].items()])
    table(["历史异常配置", "结果状态", "外层进程状态", "异常末行", "证据"], [(a["config"]["name"], a["status"], a.get("outer_status"),
          a.get("traceback", "未保存 traceback").strip().splitlines()[-1][:180], ref(a)) for a in attempts if a["status"] != "ok"])
    out.extend(["OOM 由 worker 捕获后正常退出，因此外层 ok 与结果 oom 并不矛盾；归一化数据分别保留两种状态。", ""])
    section("16. 运行登记与费用：不完整小计，不是账单", "按 dispatch 的 GPU 数纠正早期默认单卡计价。沿用实验记录费率 B200 $0.001736/GPU/s，8 核 CPU+64 GiB RAM $0.00024688/s。仅 summary.seconds×费率，缺 summary、启动/导入/镜像、存储、CPU 附件任务均不在内。")
    known = {r["run_id"]: r for r in costs["runs"]}; rows = []
    for r in runs:
        k = known.get(r["run_id"], {})
        rows.append((r["run_code"], r["run_id"], r["dispatch"]["gpu_count"], f"{sum(a['run_id']==r['run_id'] for a in attempts)}/{len(r['requested_cases'])}", fmt(k.get("runtime_seconds")), fmt(k.get("corrected_runtime_estimate_usd"), 5), "有" if k else "缺失"))
    table(["R", "run_id", "GPU", "已有/请求 case", "运行秒", "估价 USD", "summary"], rows)
    out.extend([f"已知 summary 的运行费率估价小计 ${costs['total_corrected_runtime_estimate_usd']:.5f}，覆盖 {len(costs['runs'])}/{len(runs)} 个已登记 GPU job。"
        + (f"另有 {len(costs['missing_summary_runs'])} 个 job 缺 summary、成本未知。" if costs['missing_summary_runs'] else "GPU job 的 summary 已齐；其他计费组成仍未计入，不能称总花费或账单。"), "",
        "完整来源路径、JSON pointer、配置、源码 SHA256、下载附件可用性见 evidence_index.json 和 run_registry.json。原始 runs 文件未修改。数值均可从 all_attempts.json 重算。", ""])
    (TABLES / "EXPERIMENT_TABLES.md").write_text("\n".join(out))


def main():
    DATA.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    attempts, runs = load()
    tests = [{"source": rel(p), **junit(p)} for p in sorted((ROOT / "runs/local").rglob("*.xml"))]
    costs, matrix = ledger(runs), matrices(attempts, tests)
    checks = [c for a in attempts for c in a["sample_validation"]["checks"]]
    files = [p for r in runs for p in r["local_files"]]
    coverage = {"generated_at": datetime.datetime.now(datetime.UTC).isoformat(), "attempts": len(attempts),
        "statuses": dict(Counter(a["status"] for a in attempts)), "outer_statuses": dict(Counter(a.get("outer_status") for a in attempts)),
        "completed_jobs_downloaded": sum(bool(r.get("summary")) for r in runs), "runtime_cost_estimate_usd": costs["total_corrected_runtime_estimate_usd"],
        "cost_is_incomplete_subtotal_not_invoice": True,
        "matrices": {k: {"expected": len(rows), "statuses": dict(Counter(r["status"] for r in rows))} for k, rows in matrix.items()},
        "sample_audit": {"statistic_objects": len(checks), "issues": sum(not c["count_ok"] or not c["statistics_ok"] for c in checks),
                         "do_bench_cases": sum(bool(a["config"].get("triton_bench")) for a in attempts),
                         "nonfinite_do_bench_cases": sum(not a["sample_validation"]["all_three_do_bench_means_finite_positive"] for a in attempts),
                         "cross_rank_inconsistencies": sum(not a.get("derived_metrics", {}).get("rank_max_matches_raw", True) or not a.get("derived_metrics", {}).get("bus_bandwidth_matches_raw", True) for a in attempts)},
        "artifacts": {"memory_snapshots": sum(p.endswith(".pickle") for p in files), "nsight_sqlite": sum(p.endswith(".sqlite") for p in files), "nsight_reports": sum(p.endswith(".nsys-rep") for p in files),
                      "verified_gpu_profile_cases": sum(a.get("profiling_evidence", {}).get("status") == "ok" for a in attempts),
                      "memoryviz_main_screenshots": sum(p.exists() for p in (ROOT / "output/memory/forward-memoryviz.png", ROOT / "output/memory/train-memoryviz.png"))}}
    coverage["profiling_required_configurations"] = [{"size": size, "seq": seq,
        "verified_gpu_kernel_source_indices": [a["source_index"] for a in attempts if a["config"].get("size") == size
            and a["config"].get("seq") == seq and a["config"].get("kind") == "model"
            and a.get("profiling_evidence", {}).get("status") == "ok"]}
        for size, seqs in (("small", (256, 1024, 4096)), ("xl", (256, 512, 1024))) for seq in seqs]
    dump("all_attempts.json", attempts); dump("cost_ledger.json", costs); dump("coverage_snapshot.json", coverage)
    dump("required_matrix.json", matrix); dump("local_test_history.json", tests); dump("run_registry.json", runs)
    fields = ("source_index", "run_code", "run_id", "case_dir", "config", "status", "outer_status", "source", "source_pointer", "sources", "artifact_availability", "profiling_evidence")
    dump("evidence_index.json", [{k: a.get(k) for k in fields} for a in attempts])
    dump("sample_audit.json", [{"source_index": a["source_index"], "config": a["config"], **a["sample_validation"]} for a in attempts])
    dump("profiling_evidence.json", [{"source_index": a["source_index"], "config": a["config"],
         **a["profiling_evidence"], "analysis": read(ROOT / a["profiling_evidence"]["analysis_source"])}
         for a in attempts if a.get("profiling_evidence")])
    make_tables(attempts, runs, costs, coverage, tests)
    manifest_paths = [p for p in sorted(DATA.glob("*.json")) if p.name != "generated_manifest.json"] + [TABLES / "EXPERIMENT_TABLES.md"]
    dump("generated_manifest.json", [{"path": rel(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in manifest_paths])
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    if coverage["sample_audit"]["issues"] or coverage["sample_audit"]["nonfinite_do_bench_cases"] or coverage["sample_audit"]["cross_rank_inconsistencies"]:
        raise SystemExit("Sample audit found inconsistencies; inspect sample_audit.json.")


if __name__ == "__main__":
    main()
