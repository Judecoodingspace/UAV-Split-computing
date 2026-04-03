# Setup

## Platform

Validated on:
- Jetson Orin NX 16GB
- JetPack 6.2
- Ubuntu 22.04
- CUDA 12.6
- cuDNN 9.3
- TensorRT 10.3

## Python stack

- Python `3.10.12`
- torch `2.5.0a0+8729d72e41.nv24.08`
- torchvision `0.20.0a0+afc54f7`
- ultralytics `8.3.154`
- OpenCV `4.10`

## Environment

```bash
source ~/venvs/jetson-split/bin/activate
export PYTHONPATH=/home/nvidia/jetson_split/src:$PYTHONPATH
```

## Local assets

Required but not tracked:
- weights: `/home/nvidia/jetson_split/weights/yolov8n.pt`
- input images: `/home/nvidia/jetson_split/data/`

## Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## Notes

- Prefer `--device cuda:0` if `auto` causes parsing issues.
- Do not rely on Ultralytics auto-downloading weights.
- If `venv` creation fails with `ensurepip` missing:

```bash
sudo apt install -y python3.10-venv
```
