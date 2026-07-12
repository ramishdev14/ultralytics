# Experiment Log

## EXP-000: YOLO11s Smoke Test

**Purpose:** Verify the dataset, GPU, training loop, validation loop, and output generation.

| Setting | Value |
|---|---|
| Model | YOLO11s pretrained |
| Image size | 640 |
| Epochs | 3 |
| Batch size | 16 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Final learning-rate factor | 0.1 |
| Momentum | 0.93 |
| Weight decay | 0.0005 |
| Seed | 42 |

### Final epoch results

| Metric | Value |
|---|---:|
| Precision | 0.81978 |
| Recall | 0.75595 |
| mAP@0.5 | 0.83431 |
| mAP@0.5:0.95 | 0.72326 |

**Status:** Passed.

**Interpretation:** These are pipeline-verification results and must not be reported as final journal results.