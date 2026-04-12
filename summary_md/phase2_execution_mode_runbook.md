# Phase-2 Execution Mode Runbook

## Scope

Phase-2 fixes the action space to:

- `A0`: `full_local_y8n`
- `A1`: `split_p5_fp16`
- `A2`: `split_p5_int8`
- `A3`: `full_offload_jpeg95`
- `A4`: `small_local_proxy`
- `A5`: `small_local_true`

Core resolution is `512x640`; `A4` uses `384x480`.

Current default:

- Main phase-2 run uses `A0/A1/A2/A3/A4`
- `A5` is deferred because `weights/yolov5n.pt` is not in the repo yet

## Directory Convention

Recommended sender-side output layout:

```text
outputs/phase2_execution/
  orin_nx_maxn/
    none/phase2_detail.csv
    good/phase2_detail.csv
    medium/phase2_detail.csv
    poor/phase2_detail.csv
  orin_nx_15w/
    none/phase2_detail.csv
    good/phase2_detail.csv
    medium/phase2_detail.csv
    poor/phase2_detail.csv
  cpu_fallback/
    none/phase2_detail.csv
    good/phase2_detail.csv
    medium/phase2_detail.csv
    poor/phase2_detail.csv
```

## 1. Start the Remote Receiver

Run this on the independent strong remote node `orin_nx_maxn_remote`:

```bash
cd /home/nvidia/jetson_split
python scripts/phase2_receiver.py \
  --host 0.0.0.0 \
  --port 47001 \
  --weights-y8n /home/nvidia/jetson_split/weights/yolov8n.pt \
  --device cuda:0
```

If the remote node is not in MAXN yet, switch it first at the system level.

## 2. Apply Network Profiles

Sender side:

```bash
cd /home/nvidia/jetson_split
sudo python scripts/apply_phase2_netem.py --iface <IFACE> --profile good --role sender --apply
sudo python scripts/apply_phase2_netem.py --iface <IFACE> --profile medium --role sender --apply
sudo python scripts/apply_phase2_netem.py --iface <IFACE> --profile poor --role sender --apply
```

Receiver side:

```bash
cd /home/nvidia/jetson_split
sudo python scripts/apply_phase2_netem.py --iface <IFACE> --profile good --role receiver --apply
sudo python scripts/apply_phase2_netem.py --iface <IFACE> --profile medium --role receiver --apply
sudo python scripts/apply_phase2_netem.py --iface <IFACE> --profile poor --role receiver --apply
```

Clear qdisc after each phase or before switching interfaces:

```bash
cd /home/nvidia/jetson_split
sudo python scripts/apply_phase2_netem.py --iface <IFACE> --profile clear --apply
```

## 3. Run Local-Only Actions Once per Sender Bucket

`orin_nx_maxn`:

```bash
cd /home/nvidia/jetson_split
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_maxn \
  --sender-backend cuda:0 \
  --network-profile none \
  --actions A0 A4 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_maxn/none \
  --max-images 21 \
  --warmup 1 \
  --runs 3
```

`orin_nx_15w`:

```bash
cd /home/nvidia/jetson_split
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --network-profile none \
  --actions A0 A4 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/none \
  --max-images 21 \
  --warmup 1 \
  --runs 3
```

`cpu_fallback`:

```bash
cd /home/nvidia/jetson_split
python scripts/run_phase2_execution_suite.py \
  --sender-device-id cpu_fallback \
  --sender-backend cpu \
  --network-profile none \
  --actions A0 A4 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/cpu_fallback/none \
  --max-images 21 \
  --warmup 1 \
  --runs 3
```

Note:

- `A5` requires `weights/yolov5n.pt`.
- Current repo only has `weights/yolov8n.pt`, so the main phase-2 run should not block on `A5`.

## 4. Run Remote Actions Once per Network Profile

`orin_nx_maxn`:

```bash
cd /home/nvidia/jetson_split
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_maxn \
  --sender-backend cuda:0 \
  --network-profile good \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_maxn/good \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_maxn/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3

python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_maxn \
  --sender-backend cuda:0 \
  --network-profile medium \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_maxn/medium \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_maxn/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3

python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_maxn \
  --sender-backend cuda:0 \
  --network-profile poor \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_maxn/poor \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_maxn/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3
```

`orin_nx_15w`:

```bash
cd /home/nvidia/jetson_split
python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --network-profile good \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/good \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3

python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --network-profile medium \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/medium \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3

python scripts/run_phase2_execution_suite.py \
  --sender-device-id orin_nx_15w \
  --sender-backend cuda:0 \
  --network-profile poor \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/poor \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/orin_nx_15w/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3
```

`cpu_fallback`:

```bash
cd /home/nvidia/jetson_split
python scripts/run_phase2_execution_suite.py \
  --sender-device-id cpu_fallback \
  --sender-backend cpu \
  --network-profile good \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/cpu_fallback/good \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/cpu_fallback/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3

python scripts/run_phase2_execution_suite.py \
  --sender-device-id cpu_fallback \
  --sender-backend cpu \
  --network-profile medium \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/cpu_fallback/medium \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/cpu_fallback/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3

python scripts/run_phase2_execution_suite.py \
  --sender-device-id cpu_fallback \
  --sender-backend cpu \
  --network-profile poor \
  --receiver-device-id orin_nx_maxn_remote \
  --remote-host <REMOTE_IP> \
  --remote-port 47001 \
  --actions A1 A2 A3 \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/cpu_fallback/poor \
  --reference-detail-csv /home/nvidia/jetson_split/outputs/phase2_execution/cpu_fallback/none/phase2_detail.csv \
  --max-images 21 \
  --warmup 1 \
  --runs 3
```

## 5. Aggregate All Phase-2 Results

```bash
cd /home/nvidia/jetson_split
python scripts/build_phase2_execution_report.py \
  --suite-root /home/nvidia/jetson_split/outputs/phase2_execution \
  --output-dir /home/nvidia/jetson_split/outputs/phase2_execution/_analysis
```

Core outputs:

- `phase2_summary.csv`
- `phase2_feasibility_map.csv`
- `phase2_mode_selection_latency.csv`
- `phase2_pareto_latency_bytes.csv`
- `phase2_pareto_latency_energy.csv`
- `phase2_execution_report.md`

## 6. Recommended Execution Order

1. Start `orin_nx_maxn_remote` receiver
2. `orin_nx_maxn` local-only
3. `orin_nx_15w` local-only
4. `cpu_fallback` local-only
5. For each sender bucket, run `good -> medium -> poor`
6. Aggregate once all 12 sender-side buckets are complete
7. Inspect the mode-selection table first

## 7. Acceptance Checks

- `A1/A2/A3` must produce sender and receiver timing fields.
- `A1/A2` payload bytes should stay in the same order of magnitude as phase-1.
- `A3` on `good` should align closely with remote `yolov8n` detections.
- `A4/A5` are allowed to miss strict fidelity, but must complete stably.
