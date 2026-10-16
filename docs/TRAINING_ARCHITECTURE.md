# Model Architecture and Fine-Tuning

Updated: 2026-10-16

This document describes **the uploaded fireguard_s_local_v1 checkpoint**. Training design is referenced to the [upstream pipeline source](https://github.com/ParsaVictor/FireGuuard_Fire_detection_smoke/blob/main/build_pipeline_notebook.py); actual structure is read from our saved weights, and run parameters come from the saved stage configurations. Locally inspected implementation: Ultralytics 8.4.138.

## 1. Checkpoint identity

| Item | Value |
|---|---|
| Weight file | [best.pt](../checkpoints/fireguard_s_local_v1/final/best.pt) |
| Run | fireguard_s_local_v1 |
| Architecture / scale | YOLO26 / s |
| Classes | 0 = fire; 1 = smoke |
| Input channels | 3 |
| Parameters | 9,949,412 |
| Detection strides | 8, 16, 32 |
| End-to-end head | Enabled |
| Regression bins (reg_max) | 1 |
| File size | 20,292,286 bytes |
| SHA-256 | `2c401218a735076f48c62c2c6e92cea27da5f85b03d4e99d0f564963ff6b4dea` |

The [machine-readable architecture](model_architecture.json) includes the checkpoint's YAML and the parameter count for every top-level module. Counts describe the loaded, unfused checkpoint, including both end-to-end training branches; deployment representations can differ.

## 2. Network structure

The model uses convolutional feature extraction, bidirectional multi-scale fusion, and a two-class detection head. Fine-tuning changes learned weights and the number of classes; it does not introduce a separate custom backbone or an additional fire-verification network.

The checkpoint's s scale uses depth factor **0.5**, width factor **0.5**, and maximum-channel setting **1024**. YAML channel specifications are scaled when the network is built; the actual channels below are from the loaded model.

### Backbone: modules 0-10

| Index | Module | Output channels | Role |
|---|---|---:|---|
| 0 | Conv, stride 2 | 32 | Initial downsampling and local features |
| 1 | Conv, stride 2 | 64 | Downsampling to stride 4 |
| 2 | C3k2 | 128 | Feature refinement |
| 3 | Conv, stride 2 | 128 | Downsampling to stride 8 |
| 4 | C3k2 | 256 | P3 backbone feature |
| 5 | Conv, stride 2 | 256 | Downsampling to stride 16 |
| 6 | C3k2 | 256 | P4 backbone feature |
| 7 | Conv, stride 2 | 512 | Downsampling to stride 32 |
| 8 | C3k2 | 512 | Deep semantic features |
| 9 | SPPF | 512 | Spatial pooling and context aggregation |
| 10 | C2PSA | 512 | Attention-based feature refinement; P5 backbone feature |

Conv blocks perform convolution, normalization, and activation. C3k2 blocks combine feature transformation and partial feature reuse. SPPF broadens the spatial context through pooling; C2PSA refines features with attention.

### Neck: modules 11-22

The neck first upsamples deep features and concatenates them with backbone features, then downsamples and fuses them again. This passes semantic context toward finer maps and spatial detail toward coarser maps.

| Output node | Fusion path | Channels | Stride |
|---|---|---:|---:|
| 13 | Upsample node 10; concatenate with node 6; C3k2 | 256 | 16 |
| 16 | Upsample node 13; concatenate with node 4; C3k2 | 128 | 8 |
| 19 | Downsample node 16; concatenate with node 13; C3k2 | 256 | 16 |
| 22 | Downsample node 19; concatenate with node 10; C3k2 | 512 | 32 |

### Detection head: module 23

The Detect module takes nodes **16, 19, and 22** and predicts bounding boxes and the two class scores at three scales.

| Square input | P3 / stride 8 | P4 / stride 16 | P5 / stride 32 |
|---|---|---|---|
| 512 | 64 × 64 | 32 × 32 | 16 × 16 |
| 640 | 80 × 80 | 40 × 40 | 20 × 20 |
| 768 | 96 × 96 | 48 × 48 | 24 × 24 |
| 960 | 120 × 120 | 60 × 60 | 30 × 30 |

These are feature-map dimensions for square inputs. Rectangular inference can produce different dimensions after letterboxing. A finer feature map can help represent a small flame, but higher resolution does not guarantee a higher confidence or improved recall for every example.

### End-to-end prediction and deduplication

The checkpoint has `end2end=True`. The inspected implementation trains one-to-many and one-to-one branches and uses the one-to-one path for end-to-end inference, avoiding the conventional dense-prediction NMS workflow.

That does not guarantee that every visible smoke plume receives exactly one box. Our video processing additionally suppresses same-class overlapping/contained boxes. This is post-processing; it does not modify the network or fine-tune its weights. Different-class fire and smoke predictions are retained separately.

The checkpoint uses `reg_max=1`, rather than a multi-bin DFL regression head. The saved argument `dfl=1.5` is a generic loss setting and must not be used to claim that a multi-bin DFL head exists. Saved training CSVs label the third regression-related series `l1_loss`. The exact loss implementation depends on the Ultralytics version.

## 3. What fine-tuning does

1. Start with `yolo26s.pt` pretrained on COCO.
2. Build the fire/smoke dataset configuration with two classes.
3. Reuse compatible pretrained tensors. Class-count-dependent output tensors are resized/reinitialized as needed; it is inaccurate to imply that every head tensor must be discarded.
4. Train the backbone, fusion layers, and detection head together. Both saved configurations have `freeze: null`.
5. Save validation-selected best weights and continue the resolution schedule below.

The model is a fine-tuned detector, not a model trained from random initialization. There is no recorded frozen-backbone stage, custom attention insertion, segmentation head, or separate small-flame network.

## 4. Two-stage schedule and resume

| Setting | Stage 1: s1_lo | Stage 2: s2_hi |
|---|---|---|
| Input size | 512 | 640 |
| Epochs completed | 72 | 48 |
| Initialization | COCO-pretrained YOLO26s | Stage 1 best.pt |
| Optimization scope | Full network | Full network |
| Within-stage continuation | Saved args show resume from Stage 1 last.pt | Saved args show resume=False |
| Cross-stage optimizer state | Not applicable | New optimizer and scheduler |

The upstream schedule assigns 60% of 120 epochs to the lower resolution: 72 + 48. Smaller inputs reduce compute per training step; the second stage adapts the model to its final training resolution. A measured improvement over an all-640 run would require an ablation, which is not claimed here.

A resolution change uses a **new train call** initialized with the previous best weights. A within-stage resume instead restores training state from last.pt. These operations must not be confused: resuming the old stage can restore its old image size.

The final file is selected from the completed final stage's best checkpoint; it should not be described as necessarily the weights from the last epoch. The logs show 120 completed epochs, excluding the preliminary two-epoch smoke test.

## 5. Saved optimization settings

| Parameter | Saved value |
|---|---|
| optimizer | auto |
| lr0 / lrf | 0.001 / 0.01 |
| cos_lr | True |
| warmup_epochs | 3 |
| momentum / weight_decay | 0.937 / 0.0005 |
| warmup_momentum | 0.8 |
| warmup_bias_lr | Stage 1: 0.0; Stage 2: 0.1 |
| batch / nbs | -1 (automatic) / 64 |
| amp | True |
| patience | 40 |
| workers / cache | 0 / False |
| device | CUDA device 0 |
| seed / deterministic | 0 / True |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| val / split | True / val |
| validation IoU / max_det | 0.7 / 300 |

`optimizer=auto` lets the installed trainer select an optimizer and effective learning rate. Saved lr0/momentum values are configuration fields, not proof that the automatically selected optimizer used them unchanged. The exact optimizer type and numeric AutoBatch result require the original trainer log or optimizer state; these are not inferred from the YAML alone.

Training's validation IoU of 0.7 is separate from the later video deduplication threshold of 0.30.

## 6. Augmentation settings

| Parameter | Value |
|---|---|
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.5 |
| degrees / translate / scale | 5.0 / 0.1 / 0.5 |
| shear / perspective | 2.0 / 0.0 |
| flipud / fliplr | 0.0 / 0.5 |
| mosaic / close_mosaic | 1.0 / last 15 epochs |
| mixup | 0.10 |
| cutmix / copy_paste | 0.0 / 0.0 |
| erasing | 0.20 in config; classification-only setting, not an active detection augmentation |

Additional CCTV augmentation in both saved stages includes JPEG compression, downscaling, grayscale conversion, motion blur, brightness/contrast variation, and CLAHE. Probabilities and ranges are listed in [PLAN_V2.md](../PLAN_V2.md#3-completed-training-and-fine-tuning).

## 7. Training size versus inference size

**512 → 640** is the completed training schedule. **768 and 960** were later selected for video inference. Changing inference size does not retrain the checkpoint.

The published videos use confidence 0.25. Validation calibration in the final model card recommended 0.10 under an image-trigger criterion. These are different operating points; neither is a universal guarantee of object recall.

## 8. Load the uploaded weights

Use an Ultralytics release supporting YOLO26; local inspection used 8.4.138.

```python
from ultralytics import YOLO

model = YOLO("checkpoints/fireguard_s_local_v1/final/best.pt")
results = model.predict(
    source="path/to/video.mp4",
    imgsz=960,
    conf=0.25,
    save=True,
)
```

This example uses the standard Ultralytics prediction path. It does not reproduce our additional deduplication or processing-FPS overlay. See [fireguard_core.py](../fireguard_core.py) and the [recorded demo parameters](videos/tests/README.md) for the project's processing behavior.

## Evidence and limits

- Model YAML, class count, strides, end-to-end flag, module types, and parameter count: inspected directly from the uploaded checkpoint.
- Epoch counts and training settings: local s1_lo/s2_hi results.csv and args.yaml.
- Two-stage design: [upstream pipeline generator](https://github.com/ParsaVictor/FireGuuard_Fire_detection_smoke/blob/main/build_pipeline_notebook.py), adapted by [train_local.py](../train_local.py).
- Evaluation/calibration values and limitations: [PLAN_V2.md](../PLAN_V2.md).
- This documentation update inspected artifacts; it did not retrain the model or run a new accuracy or speed benchmark.
