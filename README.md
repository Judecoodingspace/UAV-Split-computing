# UAV-Split-computing

Jetson Orin NX split-YOLO experiments for building a local `prefix -> payload -> suffix` compute profile and preparing later `split + codec` research.

## What this repo does

Current scope:
- Jetson-side prefix execution
- split payload export
- suffix replay from payload
- benchmarking for `p3 / p4 / p5`
- phase-1 `split x codec` benchmark scripts
- per-device winner-map experiment tooling
- phase-2 execution-mode experiment tooling for `full_local / split / full_offload / small-model local`

This is a research and benchmarking repo, not a production inference service.

## Current status

Completed:
- Jetson split executor
- single-image and batch prefix benchmarks
- payload export
- suffix replay benchmark
- local compute profiling for `p3 / p4 / p5`

Next:
- collect real double-machine phase-2 outputs across device and network profiles
- compare execution modes under heterogeneous compute and network conditions

## Split definition

- `p3`: payload layers `[9, 12, 15]`, replay start `16`
- `p4`: payload layers `[9, 15, 18]`, replay start `19`
- `p5`: payload layers `[15, 18, 21]`, replay start `22`

## Repo layout

```text
src/         core split executor
scripts/     benchmark and export scripts
summary_md/  research handoff notes
```

Local-only directories not tracked:
- `data/`
- `weights/`
- `outputs/`

## Quick start

```bash
source ~/venvs/jetson-split/bin/activate
export PYTHONPATH=/home/nvidia/jetson_split/src:$PYTHONPATH
```

Prefix benchmark:

```bash
python scripts/benchmark_split_prefix_batch_jetson.py \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/front_baseline_batch \
  --device cuda:0 \
  --runs 20 \
  --warmup 10
```

Payload export:

```bash
python scripts/export_split_payload_jetson.py \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/payload_bank \
  --device cuda:0 \
  --splits p3 p4 p5
```

Suffix benchmark:

```bash
python scripts/benchmark_split_suffix_jetson_v2.py \
  --payload-dir /home/nvidia/jetson_split/outputs/payload_bank \
  --output-dir /home/nvidia/jetson_split/outputs/suffix_baseline_v2 \
  --device cuda:0
```

Per-device phase-1 suite:

```bash
python scripts/run_device_phase1_suite.py \
  --device-name orin_nx_15w \
  --device-label "Jetson Orin NX 15W" \
  --device cuda:0
```

Per-device winner map:

```bash
python scripts/build_device_winner_map.py \
  --device-root /home/nvidia/jetson_split/outputs/device_profiles \
  --output-dir /home/nvidia/jetson_split/outputs/device_profiles/_analysis
```

Phase-2 receiver:

```bash
python scripts/phase2_receiver.py \
  --host 0.0.0.0 \
  --port 47001 \
  --device cuda:0
```

Phase-2 local sender suite:

```bash
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --network-profile none \
  --actions A0 A4 A5 \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/none
```

Phase-2 remote sender suite:

```bash
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --network-profile good \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --actions A1 A2 A3 \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/good \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/none/phase2_detail.csv
```

Phase-2 report build:

```bash
python scripts/build_phase2_execution_report.py \
  --suite-root /home/nvidia/jetson_split/outputs/phase2_execution \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/_analysis
```

## Current findings

At `512 x 640`:
- prefix cost: `p3 < p4 < p5`
- suffix cost: `p3 > p4 > p5`
- raw payload size is the same for all three splits: `2293760 B`
- `p4` is currently the most balanced split

## Docs

- `SETUP.md`
- `USAGE.md`
- `RESULTS.md`
- `summary_md/device_winner_map_experiment_plan.md`
- `summary_md/phase2_execution_mode_runbook.md`
- `summary_md/jetson_split_handoff_summary.md`
