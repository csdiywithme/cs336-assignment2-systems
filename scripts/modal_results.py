"""Read-only cloud status and immutable artifact snapshots; never launches GPUs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath

import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dispatch = json.loads((root / "runs/dispatch" / args.run_id / "dispatch.json").read_text())
    snapshot = {"captured_at": datetime.datetime.now(datetime.UTC).isoformat(), "dispatch": dispatch}
    try:
        snapshot["result"] = modal.FunctionCall.from_id(dispatch["call_id"]).get(timeout=0)
        snapshot["status"] = "returned"
    except TimeoutError:
        snapshot["status"] = "pending"
    except Exception as exc:
        snapshot.update(status="result_read_error", error=repr(exc))
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    status_path = root / "runs/status" / (args.run_id + "-" + stamp + ".json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(snapshot, indent=2))
    if args.brief and "result" in snapshot:
        r = snapshot["result"]
        print(json.dumps({"run_id": r["run_id"], "seconds": r["seconds"], "cost_estimate": r["estimated_compute_usd"],
                          "cases": [{"name": c["name"], "status": c["status"], "seconds": c["seconds"],
                                     "result_status": c.get("result", {}).get("status"), "trace": c.get("trace_created"),
                                     "traceback": c.get("result", {}).get("traceback"),
                                     "leaderboard": c.get("result", {}).get("measurements", {}).get("rank_max_step")}
                                    for c in r["cases"]]}, indent=2))
    else:
        print(json.dumps(snapshot, indent=2))
    if not args.download:
        return
    volume = modal.Volume.from_name("cs336-assignment2-20260914")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    local = root / "runs/cloud" / args.run_id / stamp
    local.mkdir(parents=True, exist_ok=False)
    (local / "modal_status.json").write_text(json.dumps(snapshot, indent=2))
    manifest = []
    for entry in volume.iterdir(args.run_id, recursive=True):
        if entry.type.name != "FILE":
            continue
        remote = PurePosixPath(entry.path.lstrip("/"))
        relative = remote.relative_to(args.run_id)
        if ".." in relative.parts or (entry.size > 20_000_000 and not args.large):
            continue
        target = local.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        error = None
        for attempt in range(3):
            digest = hashlib.sha256()
            size = 0
            scratch = target.with_name(target.name + f".attempt{attempt}")
            try:
                with scratch.open("xb") as stream:
                    for chunk in volume.read_file(str(remote)):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                scratch.rename(target)
                error = None
                break
            except Exception as exc:
                error = repr(exc)
        if error:
            manifest.append({"path": str(relative), "error": error})
            continue
        manifest.append({"path": str(relative), "size": size, "sha256": digest.hexdigest(),
                         "listed_size": entry.size, "stable_size": size == entry.size})
    (local / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Downloaded {len(manifest)} files to {local}")


if __name__ == "__main__":
    main()
