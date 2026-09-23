"""Synthetic CPU-only regression fixtures for Nsight launch attribution."""
import hashlib
import sqlite3

from cs336_systems.nsight_summary import process_identity, summarize


def test_thread_phase_and_process_local_correlation(tmp_path):
    path = tmp_path / "trace.sqlite"
    process, other_process = 17 << 24, 18 << 24
    main_thread, worker, other_thread = process + 17, process + 29, other_process + 18
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER,
                demangledName INTEGER, streamId INTEGER, deviceId INTEGER);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
        """)
        db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "test_sgemm"), (2, "elementwise_kernel")])
        # Worker launch belongs to main-thread phase. Unrelated process reuses ID.
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                       [(120, 130, worker, 7), (140, 150, other_thread, 7), (160, 170, main_thread, 8)])
        # GPU execution after range close must still follow the CPU launch.
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [(400, 450, process, 7, 1, 1, 0), (460, 490, other_process, 7, 2, 1, 1),
                        (500, 520, process, 8, 2, 1, 0)])
        db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
                       [(100, 200, main_thread, "measurement", None), (110, 190, main_thread, "backward", None),
                        (115, 175, main_thread, "attention_softmax", None)])
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    report = summarize(path, tmp_path / "derived")
    assert process_identity(worker) == process
    assert process_identity((1 << 48) + worker) == (1 << 48) + process
    assert report["status"] == "ok"
    assert report["unmatched_launch_kernel_count"] == 0
    assert report["ranges"]["backward"]["kernel_count"] == 2
    assert report["ranges"]["measurement"]["kernel_count"] == 2
    assert abs(report["ranges"]["backward"]["gpu_kernel_ms"] - 70 / 1e6) < 1e-15
    assert report["ranges"]["attention_softmax"]["kernel_count"] == 1
    assert report["ranges"]["attention_softmax"]["kernel_summary"][0]["name"] == "elementwise_kernel"
    assert report["phase_audit"]["measurement_unassigned_count"] == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    assert not (tmp_path / "nsys_summary.json").exists()


def test_missing_gpu_kernel_table_is_explicit(tmp_path):
    path = tmp_path / "cpu_only.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT)")
        db.execute("INSERT INTO NVTX_EVENTS VALUES (0, 10, ?, 'forward')", ((1 << 24) + 1,))
    report = summarize(path)
    assert report["status"] == "unsupported_missing_gpu_kernels"
    assert report["kernel_count"] == 0
    assert report["ranges"]["forward"]["matmul_fraction"] is None
    assert not (tmp_path / "nsys_summary.json").exists()
