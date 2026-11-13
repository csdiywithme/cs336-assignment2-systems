"""Read a dual-rank FSDP2 Nsight Systems SQLite export and optional rank event data."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics

STEP = re.compile(r"cebench/rank=(\d+)/step=(\d+)$")
LAYER_EVENT = re.compile(r"cebench/layer=([^/]+)/phase=(forward|backward)/(unshard|wait)$")
PROJECTION = re.compile(r"cebench/layer=([^/]+)/gemm=([^/]+)$")
FORWARD = re.compile(r"cebench/layer=([^/]+)/forward$")
GEMM = re.compile(r"gemm|gemv|matmul|(?:x|h|w|tcgen\d*)mma", re.IGNORECASE)
KERNEL = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY"
RUNTIME = "CUPTI_ACTIVITY_KIND_RUNTIME"
SYNC = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
EVENT = "CUPTI_ACTIVITY_KIND_CUDA_EVENT"


def process_id(global_tid):
    # Nsight serializes the low 24 bits as the thread ID and clears them for globalPid.
    return (global_tid >> 24) << 24


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def payload_fields(raw):
    """Preserve integer NCCL identifiers without lossy floating point conversions."""
    if not raw:
        return {}
    try:
        root = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    result = defaultdict(set)

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, (str, int)):
                    result[normalized(key)].add(str(child))
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(root)
    return result


class EventIndex:
    def __init__(self, events, key):
        self.groups = defaultdict(list)
        for event in events:
            self.groups[key(event)].append(event)
        self.starts = {}
        for group_key, rows in self.groups.items():
            rows.sort(key=lambda row: row["start"])
            self.starts[group_key] = [row["start"] for row in rows]

    def within(self, key, start, end):
        rows = self.groups.get(key, ())
        starts = self.starts.get(key, ())
        return rows[bisect_left(starts, start):bisect_left(starts, end)]


def children(index, parent):
    return [
        row for row in index.within(parent["globalTid"], parent["start"], parent["end"])
        if row["end"] <= parent["end"]
    ]


def time_info(events):
    if not events:
        return None
    return {
        "start_ns": min(row["start"] for row in events),
        "end_ns": max(row["end"] for row in events),
    }


def union_duration(intervals):
    total = 0
    boundary = None
    for start, end in sorted(intervals):
        uncovered = start if boundary is None else max(start, boundary)
        if end > uncovered:
            total += end - uncovered
            boundary = end
    return total


def is_allgather(nvtx):
    name = normalized(nvtx["name"]).removeprefix("nccl")
    return name == "allgather" or name.startswith("allgatherapi")


def is_ce_collective(nvtx):
    name = normalized(nvtx["name"])
    # Nsight Systems 2026.5 names the CE parent AllGather, the same as its
    # NCCL API range. The CE parent has CeSeqNum and SyncStrategy in its payload.
    if {"ceseqnum", "syncstrategy"} <= nvtx["payload"].keys():
        return True
    tags = set().union(
        *(nvtx["payload"].get(key, set()) for key in ("eventtype", "eventname", "type"))
    )
    return any(
        token in name or any(token in normalized(tag) for tag in tags)
        for token in ("cecollective", "cecoll", "copyenginecollective")
    ) or name in {"ceallgather", "allgatherce"}


def same_api(left, right):
    return bool(left["payload"].get("apiid", set()) & right["payload"].get("apiid", set()))


def same_api_group(left, right):
    return (
        left["globalTid"] == right["globalTid"]
        and bool(left["payload"].get("apigroupid", set()) & right["payload"].get("apigroupid", set()))
    )


def gpu_ref(row):
    return {key: row[key] for key in ("start", "end", "streamId", "deviceId", "correlationId", "name")}


def cpu_ref(row):
    result = {key: row[key] for key in ("start", "end", "globalTid", "name")}
    if "eventClass" in row:
        result["event_class"] = row["eventClass"]
    return result


def nccl_ref(row):
    return {**cpu_ref(row), "event_id": row.get("eventId"),
            "payload": {key: sorted(values) for key, values in row["payload"].items()}}


class Trace:
    def __init__(self, db):
        self.db = db
        db.row_factory = sqlite3.Row
        self.tables = {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing = {"StringIds", "NVTX_EVENTS", KERNEL, RUNTIME} - self.tables
        if missing:
            raise ValueError(f"SQLite export is missing required tables: {sorted(missing)}")

        self.strings = dict(db.execute("SELECT id, value FROM StringIds"))
        nvtx = self.rows("NVTX_EVENTS")
        self.domains = {
            (process_id(row["globalTid"]), row["domainId"]): self.nvtx_name(row)
            for row in nvtx if row["eventType"] == 75 and row["globalTid"] is not None
        }
        self.nvtx = []
        for row in nvtx:
            if row["end"] is None or row["globalTid"] is None:
                continue
            row["pid"] = process_id(row["globalTid"])
            row["name"] = self.nvtx_name(row)
            row["domain"] = self.domains.get((row["pid"], row["domainId"]), "")
            self.nvtx.append(row)
        self.nvtx_processes = EventIndex(self.nvtx, lambda row: row["pid"])
        self.nccl = [row for row in self.nvtx if "nccl" in row["domain"].lower()]
        for row in self.nccl:
            row["payload"] = payload_fields(row.get("jsonText") or self.strings.get(row.get("jsonTextId")))
        self.nccl_processes = EventIndex(self.nccl, lambda row: row["pid"])

        runtime = [row for row in self.rows(RUNTIME) if row["globalTid"] is not None]
        # Nsight exports both CUDA runtime and driver calls in this table
        # (eventClass 0 and 1 respectively), including batched memcpy launches.
        for row in runtime:
            row["pid"] = process_id(row["globalTid"])
            row["name"] = self.strings.get(row["nameId"], "")
        self.runtime_threads = EventIndex(runtime, lambda row: row["globalTid"])
        self.gpu_by_api = defaultdict(list)
        self.kernels = self.rows(KERNEL)
        self.copies = self.rows(MEMCPY)
        self.device_copy_kinds = {
            row["id"] for row in self.rows("ENUM_CUDA_MEMCPY_OPER")
            if row["name"] in {"CUDA_MEMCPY_KIND_DTOD", "CUDA_MEMCPY_KIND_PTOP"}
        } or {8, 10}
        for kernel in self.kernels:
            kernel["name"] = self.strings.get(kernel.get("demangledName")) or self.strings.get(kernel["shortName"], "")
        for index, copy in enumerate(self.copies):
            copy["name"] = "memcpy"
            copy["row_index"] = index
        for row in self.kernels + self.copies:
            self.gpu_by_api[row["globalPid"], row["correlationId"]].append(row)

        self.wait_by_api = {
            (row["globalPid"], row["correlationId"]): row
            for row in self.rows(SYNC) if row.get("syncType", row.get("deprecatedSyncType")) == 2
        }
        self.record_by_sync = {
            (row["globalPid"], row["contextId"], row["eventSyncId"]): row
            for row in self.rows(EVENT) if row.get("eventSyncId")
        }
        self.kernel_processes = EventIndex(self.kernels, lambda row: row["globalPid"])

    def rows(self, table):
        if table not in self.tables:
            return []
        return [dict(row) for row in self.db.execute(f"SELECT * FROM {table}")]

    def nvtx_name(self, row):
        return row.get("text") or self.strings.get(row.get("textId"), "")

    def activities(self, scope):
        for api in children(self.runtime_threads, scope):
            for gpu in self.gpu_by_api[api["pid"], api["correlationId"]]:
                yield api, gpu

    def gather(self, launch):
        nccl = self.nccl_processes.within(launch["pid"], launch["start"], launch["end"])
        ce_rows = [row for row in nccl if is_ce_collective(row)]
        apis = [
            row for row in nccl if is_allgather(row) and not is_ce_collective(row)
            and row["globalTid"] == launch["globalTid"]
        ]
        other_calls = {
            "allreduce", "reducescatter", "alltoall", "gather", "scatter", "send", "recv",
        }
        unambiguous_scope = len(apis) == 1 and len(ce_rows) == 1 and not any(
            normalized(row["name"]).removeprefix("nccl") in other_calls for row in nccl
        )
        collectives = []
        for ce in ce_rows:
            labels_allgather = any(
                "allgather" in normalized(label)
                for label in [ce["name"], *set().union(*ce["payload"].values())]
            )
            # The marked FSDP call submits one NCCL AllGather. Match the plugin API
            # ID/group or use same-thread containment when the export lacks IDs.
            linked_api = len(apis) == 1 and (
                same_api(apis[0], ce) or same_api_group(apis[0], ce) or (
                    ce["globalTid"] == apis[0]["globalTid"]
                    and apis[0]["start"] <= ce["start"] and ce["end"] <= apis[0]["end"]
                )
            )
            # GroupLaunch may be a sibling of the API range. It is still bounded
            # by the single-all-gather FSDP scope on the application thread.
            unique_in_scope = unambiguous_scope and ce["globalTid"] == launch["globalTid"]
            if linked_api or unique_in_scope or (
                len(apis) <= 1 and ce["globalTid"] == launch["globalTid"] and labels_allgather
            ):
                collectives.append(ce)

        copy_apis = {}
        for ce in collectives:
            for api, copy in self.activities(ce):
                # NCCL 2.29.7 CE sync emits 4-byte host-to-device flags.
                # The CE payload uses device copies, including batched copies.
                if (
                    copy.get("bytes", 0) > 0 and copy.get("copyKind") in self.device_copy_kinds
                    and "memcpy" in api["name"].lower()
                ):
                    # One API can emit multiple rows for batched transfers. Keep
                    # every row even when two transfers have identical timestamps.
                    copy_apis[copy["row_index"]] = (api, copy)
        copies = [copy for _, copy in copy_apis.values()]
        sm_kernels = [
            gpu for _, gpu in self.activities(launch)
            if "nccl" in gpu["name"].lower() and "allgather" in normalized(gpu["name"])
        ]
        if copies and sm_kernels:
            route = "mixed"
        elif copies:
            route = "copy_engine"
        elif sm_kernels:
            route = "sm_kernel"
        else:
            route = "unresolved"
        intervals = [(row["start"], row["end"]) for row in copies + sm_kernels]
        return {
            "route": route,
            "unshard": cpu_ref(launch),
            "nccl_allgather_api": [nccl_ref(row) for row in apis],
            "ce_ranges": [nccl_ref(row) for row in collectives],
            "unassociated_ce_ranges": [nccl_ref(row) for row in ce_rows if row not in collectives],
            "copy_ops": [
                {**gpu_ref(copy), "api": cpu_ref(api), "bytes": copy["bytes"], "count": copy.get("copyCount"),
                 "copy_kind": copy["copyKind"], "src_device": copy.get("srcDeviceId"),
                 "dst_device": copy.get("dstDeviceId")}
                for api, copy in copy_apis.values()
            ],
            "sm_kernels": [gpu_ref(row) for row in sm_kernels],
            "gpu_execution": time_info(copies + sm_kernels),
            "gpu_active_ns": union_duration(intervals),
            "device_copy_bytes": sum(row["bytes"] for row in copies),
            "intervals": intervals,
        }

    def native_waits(self, scope):
        waits = []
        for api in children(self.runtime_threads, scope):
            if "streamwaitevent" not in api["name"].lower():
                continue
            wait = self.wait_by_api.get((api["pid"], api["correlationId"]))
            row = {"api": cpu_ref(api), "correlation_id": api["correlationId"], "producer": None}
            if wait is not None:
                row["wait_stream"] = wait["streamId"]
                row["event_sync_id"] = wait.get("eventSyncId")
                record = self.record_by_sync.get((wait["globalPid"], wait["contextId"], wait.get("eventSyncId")))
                if record is not None:
                    row["producer"] = {
                        "stream": record["streamId"],
                        "gpu_timestamp_ns": record.get("timestamp") or None,
                        "correlation_id": record["correlationId"],
                    }
            waits.append(row)
        return waits


def rank_data(rank_dir, rank):
    path = rank_dir / f"rank{rank}.json"
    if not path.is_file():
        return {}, {}
    report = json.loads(path.read_text())
    waits = {}
    for row in report.get("waits", ()):
        key = (row["step"], str(row["layer"]), row["phase"])
        if key in waits:
            raise ValueError(f"Duplicate wait markers for rank {rank}, {key}")
        waits[key] = row
    meta = {key: report.get(key) for key in ("workload", "variant", "forward_prefetch")}
    meta["profiled_step_duration_ms"] = {row["step"]: row["duration_ms"] for row in report.get("samples", ())}
    return meta, waits


def step_report(trace, marker, step, measured_waits):
    launches = defaultdict(list)
    waits = defaultdict(list)
    forward = {}
    projections = []
    scopes = trace.nvtx_processes.within(marker["pid"], marker["start"], marker["end"])
    for row in scopes:
        if row["end"] > marker["end"]:
            continue
        layer_event = LAYER_EVENT.fullmatch(row["name"])
        projection = PROJECTION.fullmatch(row["name"])
        forward_layer = FORWARD.fullmatch(row["name"])
        if layer_event:
            layer, phase, kind = layer_event.groups()
            (launches if kind == "unshard" else waits)[layer, phase].append(row)
        elif projection:
            projections.append((*projection.groups(), row))
        elif forward_layer:
            forward[forward_layer[1]] = row

    layer_rows = []
    communication = []
    for layer, phase in sorted(set(launches) | set(waits)):
        gathers = [trace.gather(launch) for launch in launches[layer, phase]]
        for gather in gathers:
            communication.extend(gather.pop("intervals"))
        consumers = []
        if phase == "forward" and layer in forward:
            consumers = [gpu for _, gpu in trace.activities(forward[layer]) if gpu.get("shortName")]
        first = min(consumers, key=lambda row: row["start"], default=None)
        native_waits = [wait for scope in waits[layer, phase] for wait in trace.native_waits(scope)]
        gather_ends = [gather["gpu_execution"]["end_ns"] for gather in gathers if gather["gpu_execution"]]
        consumer_gap = first["start"] - max(gather_ends) if first is not None and gather_ends else None
        layer_rows.append({
            "layer": layer,
            "phase": phase,
            "gathers": gathers,
            "native_waits": native_waits,
            "measured_wait": measured_waits.get((step, layer, phase)),
            "first_forward_consumer": gpu_ref(first) if first is not None else None,
            "gpu_gather_end_to_first_forward_consumer_ns": consumer_gap,
        })

    gemms = []
    ordinals = Counter()
    for layer, op, projection in projections:
        kernels = sorted(
            (
                gpu for _, gpu in trace.activities(projection)
                if gpu.get("shortName") and GEMM.search(gpu["name"])
            ),
            key=lambda row: row["start"],
        )
        for gpu in kernels:
            start, end = gpu["start"], gpu["end"]
            ordinal = ordinals[layer, op]
            ordinals[layer, op] += 1
            overlap = union_duration((max(start, s), min(end, e)) for s, e in communication if s < end and e > start)
            gemms.append({
                **gpu_ref(gpu), "layer": layer, "op": op, "ordinal": ordinal,
                "grid": [gpu.get(f"grid{axis}") for axis in "XYZ"],
                "block": [gpu.get(f"block{axis}") for axis in "XYZ"],
                "duration_ns": end - start, "allgather_overlap_ns": overlap,
            })
    # Training is synchronized before the step marker closes. Require GPU activity
    # inside this step, not merely activity in the process at some earlier time.
    kernels = [
        row for row in trace.kernel_processes.within(marker["pid"], marker["start"], marker["end"])
        if row["end"] <= marker["end"]
    ]
    # Fold together wait intervals if several sampled dependencies overlap.
    wait_intervals = [
        (row["measured_wait"]["compute_ready_ms"], row["measured_wait"]["gather_ready_ms"])
        for row in layer_rows if row["measured_wait"] is not None
        and "compute_ready_ms" in row["measured_wait"] and "gather_ready_ms" in row["measured_wait"]
    ]
    return {
        "step": step, "layers": layer_rows, "gemms": gemms, "kernel_count": len(kernels),
        "measured_wait_union_ms": union_duration(wait_intervals) if wait_intervals else None,
        "devices": sorted({row["deviceId"] for row in kernels}),
        "cpu_marker": cpu_ref(marker),
    }


def distribution(values):
    return {
        "count": len(values), "median": statistics.median(values) if values else None,
        "min": min(values, default=None), "max": max(values, default=None),
    }


def compare(current, baseline):
    def gemms(report):
        return {
            (rank, step["step"], row["layer"], row["op"], row["ordinal"]): row
            for rank, device in report["ranks"].items()
            for step in device["steps"] for row in step["gemms"]
        }

    def waits(report):
        return {
            (rank, step["step"], row["layer"], row["phase"]): row["measured_wait"]["exposed_wait_ms"]
            for rank, device in report["ranks"].items()
            for step in device["steps"] for row in step["layers"] if row["measured_wait"] is not None
        }

    for rank, candidate_device in current["ranks"].items():
        baseline_device = baseline["ranks"].get(rank)
        if baseline_device is not None and baseline_device.get("workload") != candidate_device.get("workload"):
            raise ValueError(f"Cannot compare rank {rank} across different workloads")

    reference = gemms(baseline)
    candidate = gemms(current)
    pairs = []
    mismatched = 0
    for key in sorted(reference.keys() & candidate.keys()):
        before, after = reference[key], candidate[key]
        signature = ("name", "grid", "block")
        if any(before[field] != after[field] for field in signature):
            mismatched += 1
            continue
        pairs.append({
            "rank": key[0], "step": key[1], "layer": key[2], "op": key[3], "ordinal": key[4],
            "baseline_ns": before["duration_ns"], "candidate_ns": after["duration_ns"],
            "delta_ns": after["duration_ns"] - before["duration_ns"],
            "baseline_overlap_ns": before["allgather_overlap_ns"],
            "candidate_overlap_ns": after["allgather_overlap_ns"],
        })
    overlapping = [pair["delta_ns"] for pair in pairs if pair["baseline_overlap_ns"] or pair["candidate_overlap_ns"]]
    reference_waits, candidate_waits = waits(baseline), waits(current)
    wait_pairs = []
    for key in sorted(reference_waits.keys() & candidate_waits.keys()):
        rank, step, layer, phase = key
        wait_pairs.append({
            "rank": rank, "step": step, "layer": layer, "phase": phase,
            "baseline_ms": reference_waits[key], "candidate_ms": candidate_waits[key],
            "delta_ms": candidate_waits[key] - reference_waits[key],
        })
    return {
        "matching": "rank, step, layer, projection and launch order; require same kernel and launch geometry",
        "pairs": pairs, "baseline_only": len(reference.keys() - candidate.keys()),
        "candidate_only": len(candidate.keys() - reference.keys()), "kernel_mismatches": mismatched,
        "overlapping_delta_ns": distribution(overlapping),
        "wait_pairs": wait_pairs,
        "wait_delta_ms": distribution([pair["delta_ms"] for pair in wait_pairs]),
    }


def analyze(sqlite_path, rank_dir, expected_ranks):
    with sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        trace = Trace(db)
    markers = defaultdict(dict)
    pid_by_rank = {}
    for row in trace.nvtx:
        match = STEP.fullmatch(row["name"])
        if match is None:
            continue
        rank, step = map(int, match.groups())
        if step in markers[rank] or (rank in pid_by_rank and row["pid"] != pid_by_rank[rank]):
            raise ValueError(f"Conflicting capture markers for rank {rank}, step {step}")
        pid_by_rank[rank] = row["pid"]
        markers[rank][step] = row

    result = {
        "sqlite": str(sqlite_path), "available_tables": sorted(trace.tables), "ranks": {},
        "notes": [
            "CE proof requires an NCCL CE range for an instrumented AllGather and a CUDA device copy "
            "joined to a memcpy API launched within that CE range. Batched copies may include both local and peer "
            "transfers. The byte total is observed device-copy volume, rather than measured NVLink traffic.",
            "The collective GPU activity interval ends with the last correlated transfer or kernel. CE stream "
            "synchronization may follow data movement. Use the event dependency and readiness markers for waiting.",
            "Measured wait is the per-rank CUDA event gap from compute-ready to gather-ready, clipped at zero. "
            "Host delay before the compute-ready event is queued is outside the measurement. "
            "Native wait links and the first forward consumer are shown separately when captured.",
            "GEMM overlap intersects each forward projection kernel with proven all-gather GPU activity on "
            "the same rank. Durations describe the profiled workload and do not establish hardware contention.",
        ],
    }
    for rank, steps in sorted(markers.items()):
        meta, waits = rank_data(rank_dir, rank)
        reports = [step_report(trace, marker, step, waits) for step, marker in sorted(steps.items())]
        layers = [layer for report in reports for layer in report["layers"]]
        routes = Counter(gather["route"] for layer in layers for gather in layer["gathers"])
        result["ranks"][str(rank)] = {
            "process_id": pid_by_rank[rank], "devices": sorted({d for report in reports for d in report["devices"]}),
            **meta, "gather_routes": dict(routes), "steps": reports,
            "exposed_wait_ms": {
                phase: distribution([
                    layer["measured_wait"]["exposed_wait_ms"] for layer in layers
                    if layer["phase"] == phase and layer["measured_wait"] is not None
                ])
                for phase in ("forward", "backward")
            },
            "unmatched_wait_records": len(waits) - sum(layer["measured_wait"] is not None for layer in layers),
            "unmatched_wait_scopes": sum(
                layer["measured_wait"] is None and bool(layer["native_waits"]) for layer in layers
            ),
        }

    coverage = list(result["ranks"].values())
    result["two_rank_gpu_coverage"] = (
        set(markers) == set(range(expected_ranks))
        and len(set(pid_by_rank.values())) == expected_ranks
        and all(all(step["kernel_count"] for step in rank["steps"]) for rank in coverage)
        and all(len(rank["devices"]) == 1 for rank in coverage)
        and len({rank["devices"][0] for rank in coverage}) == expected_ranks
    )
    result["copy_engine_confirmed_all_ranks"] = result["two_rank_gpu_coverage"] and all(
        rank["gather_routes"].get("copy_engine", 0) > 0 for rank in coverage
    )
    result["copy_engine_only_all_gathers"] = result["copy_engine_confirmed_all_ranks"] and all(
        set(rank["gather_routes"]) == {"copy_engine"} for rank in coverage
    )
    profiled_steps = sorted({step for rank in coverage for step in rank.get("profiled_step_duration_ms", {})})
    result["slowest_profiled_rank_by_step"] = {}
    for step in profiled_steps:
        durations = {
            rank: row["profiled_step_duration_ms"][step] for rank, row in result["ranks"].items()
            if step in row.get("profiled_step_duration_ms", {})
        }
        if len(durations) == expected_ranks:
            slowest = max(durations, key=durations.get)
            result["slowest_profiled_rank_by_step"][step] = {"rank": int(slowest), "duration_ms": durations[slowest]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-dir", type=Path, help="Directory containing rank<N>.json; defaults to SQLite directory")
    parser.add_argument("--baseline", type=Path, help="Default-variant trace JSON to compare matched forward GEMMs")
    parser.add_argument("--require-ce", action="store_true", help="Require positively matched CE AllGather on both ranks")
    args = parser.parse_args()
    result = analyze(args.sqlite, args.rank_dir or args.sqlite.parent, expected_ranks=2)
    if args.baseline is not None:
        result["baseline_comparison"] = compare(result, json.loads(args.baseline.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for rank, report in result["ranks"].items():
        print(f"rank {rank} devices {report['devices']}: {report['gather_routes']}")
    if not result["two_rank_gpu_coverage"]:
        parser.exit(2, "Missing distinct GPU activity or measurement marker for one or both ranks\n")
    if args.require_ce and not result["copy_engine_confirmed_all_ranks"]:
        parser.exit(2, "NCCL CE AllGather with associated GPU copy was not proven on both ranks\n")


if __name__ == "__main__":
    main()
