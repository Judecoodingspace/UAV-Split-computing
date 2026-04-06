# Device Winner Map Experiment Plan

## Goal

Build winner maps per UAV device profile instead of collapsing all measurements into one global table.

This lets us answer two questions separately:

1. Under each device profile, which action is fastest if we only optimize local time.
2. Under each device profile, which actions remain competitive when we also care about payload size and strict consistency.

## Standard Output Layout

Run each device into its own phase-1 folder:

```text
outputs/device_profiles/
  <device-id>/
    phase1/
      device_run_manifest.json
      baseline_384x480/
      baseline_512x640/
      baseline_640x640/
      detection_consistency_384x480/
      detection_consistency_512x640/
      detection_consistency_640x640/
      full_local_384x480/
      full_local_512x640/
      full_local_640x640/
```

## Step 1: Run the Phase-1 Suite on Each Device

Example:

```bash
python scripts/run_device_phase1_suite.py \
  --device-name orin_nx_15w \
  --device-label "Jetson Orin NX 15W" \
  --device cuda:0 \
  --image-dir /home/nvidia/jetson_split/data \
  --weights /home/nvidia/jetson_split/weights/yolov8n.pt \
  --resolutions 384x480 512x640 640x640 \
  --splits p3 p4 p5 \
  --codecs fp16 int8 int4 \
  --max-images 21
```

Repeat on every target profile, for example:

- `orin_nx_15w`
- `orin_nx_maxn`
- `xavier_nx_15w`
- `cpu_fallback`

## Step 2: Aggregate Across Devices

After each device has its own `phase1/` folder:

```bash
python scripts/build_device_winner_map.py \
  --device-root /home/nvidia/jetson_split/outputs/device_profiles \
  --output-dir /home/nvidia/jetson_split/outputs/device_profiles/_analysis
```

This generates:

- `device_candidates_merged.csv`
- `device_winner_map_global_fastest.csv`
- `device_winner_map_split_strict_fastest.csv`
- `device_winner_map_split_strict_smallest_payload.csv`
- `device_winner_map_split_strict_pareto.csv`
- `device_winner_map_summary.md`

## How to Read the Results

### 1. Global Fastest

This answers:

> If a device only optimizes local end-to-end time, what wins?

If every `(device, resolution)` pair picks the same action, then the policy is still collapsing under a time-only metric.

### 2. Strict Fastest Split

This answers:

> Among split candidates that pass strict consistency, which one is fastest per device?

This is the cleanest device-specific baseline map.

### 3. Strict Smallest Payload

This answers:

> Among strict-pass candidates, which one minimizes payload per device?

This is the better map for bandwidth-limited deployment.

### 4. Strict Pareto Frontier

This answers:

> Which candidates are non-dominated on `(latency, payload)` under strict consistency?

This view is the most important one if we want to avoid a trivial “time-only winner takes all” conclusion.

## Recommended Device Profiles

At minimum, build three profiles:

- light compute / low power
- medium compute
- high compute

If possible, also vary:

- thermal mode
- battery or power budget
- CPU-only fallback

## Practical Recommendation

If the time-only map still collapses to one action, do not force RL on top of it yet.

Use the per-device Pareto frontier as the action set first, then move to:

- contextual bandit
- constrained scheduling
- multi-objective policy selection

That gives us a much healthier action space than using a single global winner.
