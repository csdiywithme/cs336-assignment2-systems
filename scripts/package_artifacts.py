"""Create local handoff bundles using explicit allowlists; never uploads anything."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}


def safe_files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()
                  and not any(part in EXCLUDED_PARTS or "cache" in part.lower()
                              for part in p.relative_to(folder).parts))


def package(target, paths):
    manifest = []
    paths = sorted(set(paths))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in paths:
            name = str(p.relative_to(ROOT))
            if p.is_symlink() or p.resolve().is_relative_to(ROOT) is False:
                raise ValueError(f"Refusing path outside workspace: {name}")
            data = p.read_bytes()
            archive.writestr(name, data)
            manifest.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps({
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "Existing evidence only; see GPU_REMAINING_TASKS.md for uncompleted requirements.",
            "files": manifest,
        }, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC check failed")
    return {"path": str(target.relative_to(ROOT)), "file_count": len(manifest), "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def main():
    out = ROOT / "output/bundles"
    out.mkdir(parents=True, exist_ok=True)
    code = []
    for name in ["cs336_systems", "cs336-basics", "tests", "scripts"]:
        code.extend(safe_files(ROOT / name))
    for name in ["pyproject.toml", "uv.lock", "README.md", "LICENSE", "LICENSE.md", "benchmark.py",
                 "modal_assignment2.py", "modal_artifacts.py", "HANDOUT_ANSWERS.md",
                 "EXPERIMENT_LOG.md", "GPU_REMAINING_TASKS.md", "WORK_LOG.md"]:
        if (ROOT / name).is_file():
            code.append(ROOT / name)
    evidence = list(code)
    evidence += safe_files(ROOT / "runs")
    for name in ["data", "tables", "figures", "pdf", "memory", "profiling", "verification"]:
        evidence.extend(safe_files(ROOT / "output" / name))
    for name in ["REPORT.md", "FULL_REPORT.md", "README.md", "NSIGHT_EXISTING_ANALYSIS.md", "VERIFICATION.md"]:
        evidence.append(ROOT / "output" / name)
    evidence.append(ROOT / "cs336_assignment2_systems.pdf")
    manifests = [package(out / "assignment2_code.zip", code),
                 package(out / "assignment2_existing_evidence.zip", evidence)]
    (out / "bundle_checksums.json").write_text(json.dumps(manifests, indent=2))
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
