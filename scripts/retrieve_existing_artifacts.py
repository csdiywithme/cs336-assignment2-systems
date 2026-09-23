"""Read existing Volume files via storage API. No App, Function, GPU or CPU jobs.

This is intentionally separate from modal_artifacts.py, whose remote functions
allocate CPU containers. Budget-pause retrieval must NOT use those functions.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--only", default="")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--max-mb", type=float, default=25)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("Unsafe run ID")
    volume = modal.Volume.from_name("cs336-assignment2-20260914")
    entries = []
    for item in volume.iterdir(args.run_id, recursive=True):
        if item.type.name != "FILE":
            continue
        remote = PurePosixPath(item.path.lstrip("/"))
        relative = remote.relative_to(args.run_id)
        if ".." in relative.parts:
            raise ValueError("Unsafe relative path")
        if args.only and not any(pattern in str(relative) for pattern in args.only.split(",")):
            continue
        entries.append({"path": str(relative), "bytes": item.size, "remote": str(remote)})
    print(json.dumps(entries, indent=2), flush=True)
    if args.list_only:
        return
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ-storage-api")
    destination = ROOT / "runs/cloud" / args.run_id / stamp
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for entry in entries:
        row = dict(entry)
        if entry["bytes"] > args.max_mb * 1_000_000:
            row["status"] = "skipped_size_limit"
        else:
            target = destination / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            digest = hashlib.sha256()
            size = 0
            try:
                with partial.open("xb") as f:
                    for block in volume.read_file(entry["remote"]):
                        f.write(block)
                        digest.update(block)
                        size += len(block)
                if size != entry["bytes"]:
                    raise ValueError(f"Size changed: expected {entry['bytes']}, received {size}")
                partial.rename(target)
                row.update(status="downloaded", sha256=digest.hexdigest(), received_bytes=size)
            except Exception as exc:
                row.update(status="download_failed", error=repr(exc), received_bytes=size)
        manifest.append(row)
        (destination / "storage_download_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(row), flush=True)
    print(f"Existing-artifact snapshot: {destination}", flush=True)


if __name__ == "__main__":
    main()
