# Scale-Adaptive Dual-Attention YOLO11 Project Log

## Environment

- Python: 3.11.9
- PyTorch: 2.13.0+cu130
- Ultralytics: 8.4.92
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- GPU memory: 16 GB
- CUDA runtime: 13.0
- Development branch: journal-dual-attention

## 12 July 2026

- Created Python virtual environment.
- Installed Ultralytics from source in editable mode.
- Configured GitHub fork as `origin`.
- Configured official Ultralytics repository as `upstream`.
- Created `before-journal-work` Git tag.
- Audited the ROAD-SEC pseudo-labeled dataset.
- Verified 12,191 image-label pairs.
- Verified 161,587 valid object instances.
- Completed a three-epoch YOLO11s smoke test.
- Confirmed successful CUDA training and validation.