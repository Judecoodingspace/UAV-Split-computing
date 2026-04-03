# Usage

## Default benchmark convention

- input size: `512 x 640`
- device: `cuda:0`
- splits: `p3 p4 p5`

## 1. Single-image prefix benchmark

```bash
python scripts/benchmark_split_prefix_jetson.py \
  --image /home/nvidia/jetson_split/data/00191.jpg \
  --split p4 \
  --device cuda:0 \
  --runs 20 \
  --warmup 10
```

## 2. Batch prefix benchmark

```bash
python scripts/benchmark_split_prefix_batch_jetson.py \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/front_baseline_batch \
  --device cuda:0 \
  --runs 20 \
  --warmup 10
```

## 3. Export payloads

```bash
python scripts/export_split_payload_jetson.py \
  --image-dir /home/nvidia/jetson_split/data \
  --output-dir /home/nvidia/jetson_split/outputs/payload_bank \
  --device cuda:0 \
  --splits p3 p4 p5
```

## 4. Suffix replay benchmark

```bash
python scripts/benchmark_split_suffix_jetson_v2.py \
  --payload-dir /home/nvidia/jetson_split/outputs/payload_bank \
  --output-dir /home/nvidia/jetson_split/outputs/suffix_baseline_v2 \
  --device cuda:0 \
  --warmup 5 \
  --runs 5
```

## Outputs

- `outputs/front_baseline_batch/`: prefix CSVs
- `outputs/payload_bank/`: payload files and manifest
- `outputs/suffix_baseline_v2/`: suffix CSVs

## Recommended workflow

1. Run prefix batch benchmark.
2. Export payloads.
3. Run suffix replay benchmark.
4. Compare `p3 / p4 / p5` before adding codec experiments.
