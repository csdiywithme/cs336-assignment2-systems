"""CPU-only artifact extraction and genuine profiler GUI screenshots.

No GPU function is defined in this file. Remote paths are restricted to the
assignment's dedicated volume; downloads use immutable local snapshot folders.
"""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("cs336-assignment2-artifacts")
volume = modal.Volume.from_name("cs336-assignment2-20260914")
base = modal.Image.debian_slim(python_version="3.13")
gui = (base.apt_install("wget", "gnupg", "ca-certificates", "xvfb", "xauth", "xdotool", "imagemagick",
                        "openbox", "libgl1", "libglib2.0-0", "libxcb-cursor0", "libxkbcommon-x11-0", "libnss3")
       .add_local_file(ROOT / "scripts/nvidia-devtools.list", "/etc/apt/sources.list.d/nvidia-devtools.list", copy=True)
       .run_commands("wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub -O /tmp/nvidia-devtools.pub",
                     "gpg --dearmor --output /usr/share/keyrings/nvidia-devtools-keyring.gpg /tmp/nvidia-devtools.pub",
                     "apt-get update", "apt-get install -y --no-install-recommends nsight-systems")
       .env({"DISPLAY": ":99", "LIBGL_ALWAYS_SOFTWARE": "1", "QT_XCB_GL_INTEGRATION": "none"}))


def safe_path(relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Expected a concrete relative assignment artifact")
    return Path("/data").joinpath(*p.parts)


@app.function(image=base, timeout=120, memory=4096, volumes={"/data": volume})
def inventory(run_id: str):
    volume.reload()
    root = safe_path(run_id)
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
            for p in root.rglob("*") if p.is_file() and not any("cache" in part for part in p.relative_to(root).parts)]


@app.function(image=base, timeout=120, memory=4096, volumes={"/data": volume})
def chunk(relative: str, offset: int, length: int):
    if offset < 0 or not 0 < length <= 750000:
        raise ValueError("Invalid bounded chunk")
    volume.reload()
    with safe_path(relative).open("rb") as stream:
        stream.seek(offset)
        return stream.read(length)


@app.function(image=base, timeout=120, memory=4096, volumes={"/data": volume})
def inspect_sqlite(relative: str):
    import sqlite3
    volume.reload()
    db = sqlite3.connect(f"file:{safe_path(relative)}?mode=ro", uri=True)
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    result = {name: {"count": db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0],
                   "columns": [r[1] for r in db.execute(f'PRAGMA table_info("{name}")')]}
            for name in tables}
    if "DIAGNOSTIC_EVENT" in tables:
        result["diagnostic_messages"] = list(db.execute("SELECT severity,text FROM DIAGNOSTIC_EVENT"))
    return result


@app.function(image=gui, timeout=240, cpu=4, memory=16384, volumes={"/data": volume})
def screenshot_nsight(relative: str):
    import os
    import subprocess
    import time
    volume.reload()
    report = safe_path(relative)
    if report.suffix != ".nsys-rep" or not report.exists():
        raise ValueError("Expected an existing Nsight report")
    destination = report.parent / "gui"
    destination.mkdir(exist_ok=False)
    logs = (destination / "gui.log").open("w")
    xvfb = subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1920x1200x24"], stdout=logs, stderr=logs)
    time.sleep(1)
    wm = subprocess.Popen(["openbox"], stdout=logs, stderr=logs)
    executable = subprocess.run(["sh", "-c", "command -v nsys-ui"], capture_output=True, text=True).stdout.strip()
    if not executable:
        candidates = list(Path("/opt/nvidia").glob("nsight-systems/*/host-linux-x64/nsys-ui"))
        executable = str(sorted(candidates)[-1])
    process = subprocess.Popen([executable, str(report)], stdout=logs, stderr=logs)
    try:
        time.sleep(20)
        windows = subprocess.run(["xdotool", "search", "--name", "Nsight"], capture_output=True, text=True).stdout.split()
        if windows:
            subprocess.run(["xdotool", "windowactivate", "--sync", windows[-1]])
            subprocess.run(["xdotool", "windowsize", windows[-1], "1920", "1150"])
            subprocess.run(["xdotool", "windowmove", windows[-1], "0", "25"])
        time.sleep(3)
        subprocess.run(["import", "-window", "root", str(destination / "overview.png")], check=True)
        title = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True).stdout
        (destination / "metadata.json").write_text(json.dumps({"report": relative, "window_title": title,
                                                               "method": "Xvfb screenshot of genuine NVIDIA Nsight Systems GUI",
                                                               "returncode_at_capture": process.poll()}, indent=2))
    finally:
        process.terminate()
        wm.terminate()
        xvfb.terminate()
        logs.close()
        volume.commit()
    return {"relative": str((destination / "overview.png").relative_to("/data")), "title": title}


@app.local_entrypoint()
def main(run: str = "", only: str = "", render: str = "", list_only: bool = False, inspect: str = ""):
    if inspect:
        result = inspect_sqlite.remote(inspect)
        target = ROOT / "runs/nsight-schema.json"
        target.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    if render:
        print(screenshot_nsight.remote(render))
        return
    entries = inventory.remote(run)
    if list_only:
        print(json.dumps(entries, indent=2))
        return
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ-binary")
    local = ROOT / "runs/cloud" / run / stamp
    local.mkdir(parents=True, exist_ok=False)
    manifest = []
    for entry in entries:
        name = entry["path"]
        if only and only not in name:
            continue
        if not only and Path(name).suffix not in (".png", ".pickle", ".nsys-rep", ".sqlite"):
            continue
        target = local / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with target.open("xb") as stream:
            for offset in range(0, entry["bytes"], 750000):
                data = chunk.remote(run + "/" + name, offset, min(750000, entry["bytes"] - offset))
                stream.write(data)
                digest.update(data)
        if target.stat().st_size != entry["bytes"]:
            raise RuntimeError("Incomplete artifact download")
        manifest.append({**entry, "sha256": digest.hexdigest()})
        print(f"Downloaded {name}: {entry['bytes']} bytes", flush=True)
    (local / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
