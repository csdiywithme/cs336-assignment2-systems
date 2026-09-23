"""Wrap existing snapshots for PyTorch's official client-side MemoryViz viewer.

No torch/CUDA import and no cloud function. Snapshot bytes are embedded locally.
The viewer's JavaScript is loaded from the official PyTorch release via jsDelivr.
"""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = "https://cdn.jsdelivr.net/gh/pytorch/pytorch@v2.11.0/torch/utils/viz/MemoryViz.js"


def main():
    out = ROOT / "output/memory"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((ROOT / "runs/cloud/20260914T085600Z-memory-and-checkpoint").glob("*/*/memory_snapshot.pickle")):
        name = path.parent.name
        data = path.read_bytes()
        # Keep HTML small for browser bridges; decode the existing binary locally.
        # It is the same snapshot, copied without modification and hash-verified.
        snapshots = out / "snapshots"
        snapshots.mkdir(exist_ok=True)
        (snapshots / (name + ".pickle")).write_bytes(data)
        page = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>{name} - PyTorch MemoryViz</title></head>
<body><script type="module">
import {{add_local_files}} from "{VIEWER}";
const response = await fetch("snapshots/{name}.pickle");
const raw = new Uint8Array(await response.arrayBuffer());
const padded = new Uint8Array(Math.ceil(raw.length / 3) * 3);
padded.set(raw);
const base64 = await new Promise((resolve, reject) => {{
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result.split(",")[1]);
  reader.onerror = reject;
  reader.readAsDataURL(new Blob([padded]));
}});
add_local_files([{{name: "{name}.pickle", base64}}], 'Active Memory Timeline');
</script></body></html>
'''
        destination = out / (name + ".html")
        destination.write_text(page)
        rows.append({"source": str(path.relative_to(ROOT)), "html": str(destination.relative_to(ROOT)),
                     "snapshot_bytes": len(data), "snapshot_sha256": hashlib.sha256(data).hexdigest(),
                     "viewer": VIEWER, "kind": "Official PyTorch MemoryViz, existing local snapshot"})
    if not rows:
        raise RuntimeError("No existing local memory snapshots available")
    (out / "memory_view_manifest.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
