# UAV-Split-computing

Jetson Orin NX split-YOLO experiments for building:

- phase-1 local `prefix -> payload -> suffix` compute profiles
- phase-2 real Jetson-to-server execution-mode experiments

The repo is now used to study not only `split + codec`, but also **heterogeneity-aware execution mode selection** across:

- local full inference
- split inference
- full offload
- small local proxy inference

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
- phase-1 winner-map analysis across device buckets
- phase-2 local baselines for `A0/A4` on `orin_nx_maxn`, `orin_nx_15w`, and `cpu_fallback`
- phase-2 real Jetson-to-server remote execution for `A1/A2/A3`

Current research framing:
- phase-1 answered: which split/codec candidates are worth carrying forward
- phase-2 is answering: **under heterogeneous compute and network conditions, which execution mode should be selected**
- the main question is no longer only `split-point selection`; it is now **execution-mode selection**

Current next stage:
- expand phase-2 from smoke tests to fuller slices across `device x network x action`
- validate whether execution-mode switching appears under weaker devices and worse links
- introduce mission preference (`latency-first / fidelity-first / balanced`) as an explicit selector input

## Split definition

- `p3`: payload layers `[9, 12, 15]`, replay start `16`
- `p4`: payload layers `[9, 15, 18]`, replay start `19`
- `p5`: payload layers `[15, 18, 21]`, replay start `22`

## Phase-2 action definition

- `A0 full_local_y8n`: local `yolov8n @ 512x640`
- `A1 split_p5_fp16`: Jetson runs `p5` prefix, uploads `fp16` split payload, server runs suffix replay
- `A2 split_p5_int8`: Jetson runs `p5` prefix, uploads `int8` split payload, server runs suffix replay
- `A3 full_offload_jpeg95`: Jetson uploads `JPEG95` image, server runs full `yolov8n`
- `A4 small_local_proxy`: local `yolov8n @ 384x480`
- `A5 small_local_true`: reserved for `yolov5n @ 512x640`, currently deferred because `weights/yolov5n.pt` is not in the repo

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

Server-sync branch:
- `phase2-server`
- This branch only carries the phase-2 code/runbook needed for the remote server.
- Weights still need to be copied to the server manually.

## Quick start

```bash
source ~/venvs/jetson-split/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Remote server sync:

```bash
git fetch origin
git checkout phase2-server
git pull origin phase2-server
```

Prefix benchmark:

```bash
python scripts/benchmark_split_prefix_batch_jetson.py \
  --image-dir data \
  --output-dir outputs/front_baseline_batch \
  --device cuda:0 \
  --runs 20 \
  --warmup 10
```

Payload export:

```bash
python scripts/export_split_payload_jetson.py \
  --image-dir data \
  --output-dir outputs/payload_bank \
  --device cuda:0 \
  --splits p3 p4 p5
```

Suffix benchmark:

```bash
python scripts/benchmark_split_suffix_jetson_v2.py \
  --payload-dir outputs/payload_bank \
  --output-dir outputs/suffix_baseline_v2 \
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
  --device-root outputs/device_profiles \
  --output-dir outputs/device_profiles/_analysis
```

Phase-2 receiver:

```bash
python scripts/phase2_receiver.py \
  --host 0.0.0.0 \
  --port 47001 \
  --weights-y8n weights/yolov8n.pt \
  --device cuda:0
```

Phase-2 link proxy, useful when Jetson does not support `tc tbf/netem`:

```bash
python scripts/phase2_link_proxy.py \
  --listen-host 127.0.0.1 \
  --listen-port 47002 \
  --upstream-host <receiver-ip> \
  --upstream-port 47001 \
  --profile medium \
  --seed 123 \
  --log-jsonl outputs/link_proxy/medium.jsonl
```

Sender device profile preflight:

```bash
python scripts/device_profile_ctl.py \
  --profile orin_nx_15w \
  --sender-backend cuda:0
```

To switch power mode before a run, you can ask the helper to apply the profile:

```bash
python scripts/device_profile_ctl.py \
  --profile orin_nx_15w \
  --sender-backend cuda:0 \
  --apply
```

Phase-2 local sender suite:

```bash
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --sender-device-profile orin_nx_15w \
  --network-profile none \
  --actions A0 A4 \
  --image-dir data \
  --output-dir outputs/phase2_execution/orin_nx_15w/none
```

Phase-2 remote sender suite:

```bash
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --sender-device-profile orin_nx_15w \
  --network-profile good \
  --receiver-device-id server_remote \
  --remote-host <REMOTE_IP_OR_PROXY> \
  --remote-port <REMOTE_PORT_OR_PROXY_PORT> \
  --actions A1 A2 A3 \
  --image-dir data \
  --output-dir outputs/phase2_execution/orin_nx_15w/good \
  --reference-detail-csv outputs/phase2_execution/orin_nx_15w/none/phase2_detail.csv
```

If the proxy runs on the Jetson locally, point the sender to `127.0.0.1:47002`.

Phase-2 report build:

```bash
python scripts/build_phase2_execution_report.py \
  --suite-root outputs/phase2_execution \
  --output-dir outputs/phase2_execution/_analysis
```

## Server-side fine-tune

If you want to adapt `yolov8n.pt` to UAV imagery, prefer training on a server GPU rather than on the Jetson.

Why a server GPU is recommended:
- training is much heavier than inference, especially with `imgsz=960+`
- GPU memory is usually much larger and more stable than on-device training
- long training jobs are less likely to interfere with Jetson-side phase-2 experiments
- the Jetson is better kept for device-mode and end-to-end execution studies

Recommended dataset layout:

```text
data/uav_ft_v1/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Example `data.yaml`:

```yaml
path: /path/to/UAV-Split-computing/data/uav_ft_v1
train: images/train
val: images/val
test: images/test
names:
  0: car
  1: person
```

Recommended image counts for `train / val / test`:
- minimum usable split: `200-300` total images, roughly `140-210 / 30-45 / 30-45`
- more reliable split: `500` total images, roughly `350 / 75 / 75`
- stronger research split: `1000+` total images, roughly `700 / 150 / 150`

Split rules:
- split by scene / flight segment, not by adjacent frames
- keep the same class mapping across training and evaluation
- if you fine-tune on `car/person`, the resulting model uses `0=car, 1=person`

Recommended first fine-tune target:
- start with `weights/yolov8n.pt`
- keep the `YOLOv8n` architecture first for best compatibility with the current split executor and `A1/A2` path

Training command template:

```bash
source ~/venvs/jetson-split/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

yolo detect train \
  model=weights/yolov8n.pt \
  data=data/uav_ft_v1/data.yaml \
  imgsz=960 \
  epochs=150 \
  batch=16 \
  device=0 \
  workers=8 \
  patience=30 \
  project=runs/detect \
  name=uav_yolov8n_ft_960
```

After training:
- best checkpoint: `runs/detect/uav_yolov8n_ft_960/weights/best.pt`
- last checkpoint: `runs/detect/uav_yolov8n_ft_960/weights/last.pt`
- copy `best.pt` back to the Jetson, for example as `weights/yolov8n_uav_ft.pt`
- use the new weight in phase-2 with `--weights-y8n weights/yolov8n_uav_ft.pt`

Evaluation note:
- after a two-class fine-tune, use `pred_class_id=0` when evaluating `car`
- the old COCO-pretrained default `pred_class_id=2` no longer applies

## Current findings

### Phase-1

- `full_local` is still the pure-local global winner on strong Jetson profiles
- `p5+fp16` is the most stable split baseline
- `p5+int8 @ 512x640` is the strongest split challenger
- after adding device buckets, trade-offs begin to appear, especially on `512x640` and weaker devices

### Phase-2 local-only baselines

- `A4 small_local_proxy` is the utility winner on all three measured sender buckets
- but `A4` is not fidelity-preserving, so it cannot replace `A0 full_local`

### Phase-2 real remote actions

- `A1/A2/A3` have all been run successfully in a real Jetson-to-server setup
- `A1/A2` show that split execution is real and nearly lossless in fidelity
- `A2` halves `A1`'s payload and is the more realistic split candidate
- on `orin_nx_15w + good`, `A3` is much faster than `A1/A2`; the split path is currently dominated by feature transmission cost
- on `cpu_fallback + good`, `A3` already beats `A0 full_local` and approaches `A4` while preserving much higher fidelity

### Current interpretation

The strongest current evidence supports:

- heterogeneous devices should not all use the same action
- weak devices may prefer `A3 full_offload`
- strong devices may still prefer `A0 full_local`
- `A2` is the split mode most worth continuing to test

The repo is therefore moving from **split-point selection** toward **heterogeneity-aware execution-mode selection**.

## Interpreting phase-2 delays

- `tx_ms_uplink`: sender-side time from sending the request frame to receiving the receiver ack; this is an approximation of request/payload uplink delay
- `return_ms_downlink`: sender-side time from receiving the ack to receiving the final response, minus the receiver-reported processing time; this is an approximation of response return delay, not a physical one-way measurement

## Docs

- `SETUP.md`
- `USAGE.md`
- `RESULTS.md`
- `summary_md/device_winner_map_experiment_plan.md`
- `summary_md/phase2_execution_mode_runbook.md`
- `summary_md/jetson_split_handoff_summary.md`
