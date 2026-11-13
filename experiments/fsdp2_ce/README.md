# CS336 A2 · FSDP2 copy-engine all-gather

This follow-up to the course's distributed-training assignment measures
whether moving FSDP2 all-gather to a copy engine reduces GPU compute stream
waits and full training step time on two NVLink-connected GPUs.

[Dual-B300 results and measurement data](../../output/fsdp2-ce/)

## Workload and controls

All three variants run in fresh worker processes in one Modal container with
the same two GPUs of one model. The B300 path uses PyTorch 2.12.1 with CUDA
13.2. The B200 path uses PyTorch 2.12.1 with CUDA 13.0. The full workload uses the course XL model
(3,406,809,600 parameters, FP32, global batch 4, sequence length 512, vocabulary
10,000). It uses layer-level `fully_shard`, the same default prefetch policy and
resharding configuration, no activation checkpointing, and
`torch.optim.AdamW`. The same seeds and token/target batches are used across
variants. Synthetic data is sufficient for this short step benchmark.

| Variant | All-gather staging buffer | NCCL CTA policy |
| --- | --- | --- |
| `default` | FSDP2 default | DEFAULT |
| `symm` | `set_symm_mem_for_comm("NCCL")` | DEFAULT |
| `zero` | `set_symm_mem_for_comm("NCCL")` | ZERO |

The launcher pins the GPU software stack, records the installed package list,
GPU identities, NVLink and peer diagnostics, and hashes the uploaded sources.
Raw artifacts go to a run-specific directory in the existing CS336 volume.

## Reproduction

Run these commands from the Assignment 2 directory with the Modal CLI and
Python 3.10 or newer. Set `MODAL_PROFILE` to your own workspace profile if it
differs from the active one. Create the experiment volume once in that
workspace. Use the same GPU model for all stages.

```sh
export FSDP2_CE_GPU_MODEL=B300
modal volume create cs336-assignment2-20260914
mkdir -p runs/cloud

modal run --detach experiments/fsdp2_ce/modal_run.py --stage smoke
```

After the smoke job finishes, substitute the printed Run ID for
`SMOKE_RUN_ID` below. Give each download its own destination directory.

```sh
modal volume get cs336-assignment2-20260914 "$SMOKE_RUN_ID" "runs/cloud/$SMOKE_RUN_ID"
python experiments/fsdp2_ce/check_smoke.py "runs/cloud/$SMOKE_RUN_ID"
```

After the audit passes, run the three-way comparison and the separate
fixed-staging route control. The route stage checks numerics and confirms
the selected path with a profiler capture.

```sh
modal run --detach experiments/fsdp2_ce/modal_run.py --stage full
modal run --detach experiments/fsdp2_ce/modal_run.py --stage route
```

Set `FULL_RUN_ID` and `ROUTE_RUN_ID` from the launcher output. After both jobs
finish, download and analyze the results locally.

```sh
modal volume get cs336-assignment2-20260914 "$FULL_RUN_ID" "runs/cloud/$FULL_RUN_ID"
modal volume get cs336-assignment2-20260914 "$ROUTE_RUN_ID" "runs/cloud/$ROUTE_RUN_ID"
python experiments/fsdp2_ce/summarize.py "runs/cloud/$FULL_RUN_ID" --format markdown
python experiments/fsdp2_ce/summarize_traces.py "runs/cloud/$FULL_RUN_ID" --format markdown
python experiments/fsdp2_ce/summarize_route.py "runs/cloud/$ROUTE_RUN_ID" --format markdown
python experiments/fsdp2_ce/report.py --smoke "runs/cloud/$SMOKE_RUN_ID" \
  --full "runs/cloud/$FULL_RUN_ID" --output "runs/cloud/$FULL_RUN_ID/report.md"
```

The smoke audit checks
numerical parity, independently matches CE AllGather to GPU copy activity on
each rank, and reports whether the NVLink peer path was directly confirmed.
For the original B300 smoke, use `--recover-analysis` with `check_smoke.py`
and `--recover-smoke-analysis` with `report.py` to recheck the saved SQLite
after the Nsight 2026.5 event-matching fix.
Inspect `summary.json`, `runtime.json`, per-case `status.json` and logs to
diagnose failures. The full stage records one-step XL parity in every variant,
then rotates variant order across three fresh-process repetitions with five
warmup and twelve measured steps per case. It captures two measured steps per
variant in separate Nsight runs. Reanalyze a captured trace with:

```sh
python experiments/fsdp2_ce/analyze_trace.py <run-dir>/trace-zero/profile.sqlite \
  --output <run-dir>/trace-zero/reanalysis.json \
  --baseline <run-dir>/trace-default/trace.json --require-ce
```

## Measurement

- **Correctness.** Run one fixed step from identical weights and inputs.
  Reconstruct each gradient and updated weight from all shards and compare
  against the default FSDP2 variant. Keep per-tensor maximum absolute errors,
  mismatch counts, and the exact tolerances in the run's rank JSON.
- **Plain timing.** Take synchronized wall time for zeroing gradients, forward,
  loss, backward and AdamW. Synchronize before the timed step and take the
  per-step maximum of the two ranks before aggregation. Record max allocator
  allocated and reserved bytes. A separate device-wide NVML sampling pass
  cross-checks memory visible outside the PyTorch allocator.
- **GPU trace.** Attribute each captured gather using its layer and phase range,
  NCCL events, and CUDA launch correlations. Measure exposed wait from the
  compute stream's ready marker to the gather stream's ready marker, clipped
  at zero. This estimate includes trace-probe overhead and excludes the
  following parameter copy-out. When available, use the native CUDA event
  links to check the dependency. Match forward projection GEMMs by layer,
  projection, launch order, kernel name and launch geometry.
- **Attribution.** The full stage records a decision in `wait-diagnostic.json`.
  The heuristic uses median per-step exposed waits on the slowest profiled
  rank. At 1 ms or more across prefetchable forward blocks, the same container
  repeats zero-CTA with explicit next-layer prefetch. The forward trigger
  excludes the root and first block. If forward is below the threshold and
  backward meets it, the same container runs isolated and concurrent
  all-gather/GEMM with the XL block message size and Q projection shape.
  `--stage prefetch` and `--stage micro` run the respective control separately.
  Attributing a slowdown to an individual hardware resource requires
  additional measurements.
- **Route control.** NCCL 2.29.7 can automatically choose CE for registered
  Blackwell all-gathers above its count threshold on direct NVLink. To hold
  symmetric staging fixed while testing the route, run `--stage route` and
  verify both captured paths. That stage sets `NCCL_SYM_CE_THRESHOLD=1073741824`
  in both policy arms and rotates their unprofiled timing order.
