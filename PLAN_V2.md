# FireGuard: Training Record and Next Steps

Updated: 2026-10-16

This document describes the training workflow and saved results for this repository's **fireguard_s_local_v1** model. It is based on the local training entry point, dataset audit, stage configurations, training logs, final model card, and recorded video inference runs.

## 1. Model and artifacts

- Base checkpoint: **yolo26s.pt**, pretrained for COCO detection.
- Target classes: **0 = fire**, **1 = smoke**.
- Fine-tuning: adapt the detection head to two classes and train the full network; no layers are frozen.
- Saved model used for video inference: `checkpoints/fireguard_s_local_v1/final/best.pt`.
- Saved artifacts also include `best.onnx` and `model_card.json`.
- The checkpoint and datasets are stored separately from the public source repository.

The saved stage configurations contain Linux server paths. The checkpoint folder was subsequently copied into the local project. These historical paths must be adjusted when reproducing a run on another machine.

## 2. Dataset preparation

The pipeline combines **FASDD_CV** and **D-Fire**.

1. Download and extract the source datasets.
2. Preserve FASDD class IDs and swap D-Fire class IDs to match fire=0 and smoke=1.
3. Audit images and labels; isolate invalid annotations and clip boxes that cross image boundaries.
4. Remove perceptually similar images using dHash, with Hamming distance threshold 3.
5. Build approximate event groups using threshold 12, then assign groups to train, validation, and test splits.
6. Apply training augmentation after splitting.

The preparation audit recorded:

| Item | Images |
|---|---:|
| Source images | 116,653 |
| After isolating invalid labels | 116,627 |
| After perceptual deduplication | 66,785 |
| Training split | 59,676 |
| Validation split | 3,577 |
| Test split | 3,532 |

The audit isolated 26 images with invalid labels and clipped 50 boxes in 41 label files.

The requested 80/10/10 allocation applies to **groups**. Unequal group sizes produced image proportions of approximately 89.36/5.36/5.29%. Dataset grouping uses a heuristic and does not guarantee that every related event or near-duplicate was found. Perceptual similarity is also not proof that two images are identical.

These are historical preparation counts. The datasets are not included in the repository and were later removed from the local test workspace; reproducing training requires preparing them again.

See [TRAINING_REVIEW.md](TRAINING_REVIEW.md) for the preparation audit.

## 3. Completed training and fine-tuning

The saved CSV logs and final model card confirm completion of both stages:

| Stage | Input size | Completed epochs | Initialization |
|---|---:|---:|---|
| s1_lo | 512 | 72 | COCO-pretrained YOLO26s |
| s2_hi | 640 | 48 | Best checkpoint from s1_lo |
| Total | Progressive 512 to 640 | **120** | Full-network fine-tuning |

Stage 1 contains a within-stage resume from its own last checkpoint. Stage 2 starts a new training session from the Stage 1 best weights with `resume=False`; it does not continue the previous optimizer and scheduler state.

### Saved training settings

- Automatic batch sizing: `batch=-1`.
- Optimizer selection: `optimizer=auto`.
- Configured `lr0=0.001`, `lrf=0.01`, cosine learning-rate scheduling, and 3 warm-up epochs.
- Early-stopping patience: 40.
- Automatic mixed precision enabled.
- Training data-loader workers: 0.
- Training seed in saved stage configurations: 0. Dataset grouping uses seed 1337.
- Mosaic probability: 1.0; mosaic closes for the last 15 epochs.
- MixUp probability: 0.1.
- Rotation: 5 degrees; translation: 0.1; scale: 0.5; shear: 2 degrees.
- Horizontal flip probability: 0.5.

Automatic optimizer selection can override the configured initial learning rate. The configured value alone must not be reported as the effective learning rate throughout training.

### CCTV augmentation

Both saved stage configurations include:

| Augmentation | Probability | Settings |
|---|---:|---|
| JPEG compression | 0.30 | Quality 30-80 |
| Downscaling | 0.20 | Scale 0.35-0.85 |
| Grayscale | 0.12 | Three output channels |
| Motion blur | 0.10 | Kernel range 3-5 |
| Brightness and contrast | 0.20 | Limits ±0.25 |
| CLAHE | 0.02 | Clip limit 1-4 |

The two-epoch smoke test is a pipeline check and is not the final trained model. The ablation option is disabled by default; no completed ablation study is claimed here.

## 4. Saved evaluation records

### Final model-card detection metrics

The final model card records the following bounding-box metrics:

| Metric | Value |
|---|---:|
| mAP@0.50 | 68.39% |
| mAP@0.50:0.95 | 41.83% |
| Fire AP@0.50 | 69.50% |
| Smoke AP@0.50 | 67.28% |
| Precision | 73.66% |
| Recall | 58.96% |

These values are transcribed from the saved model card, not recomputed for this document. The metrics block does not label its evaluation split, so these values should not be presented as a newly verified independent test benchmark without the original evaluation report.

### Validation threshold calibration

The model card explicitly identifies the calibration split as **validation**:

| Item | Value |
|---|---:|
| Negative images | 2,490 |
| Positive images | 1,087 |
| Recommended confidence | 0.10 |
| Negative-image trigger rate | 0.763% |
| Positive-image trigger rate | 97.884% |
| Target negative-image trigger rate | At most 1% |

A trigger means **any detection in an image**, without matching class or bounding-box IoU. Positive-image trigger rate is therefore not object recall. Negative-image trigger rate is not false alarms per camera per day.

The later demo videos use confidence **0.25**, so their operating point differs from this calibration.

## 5. Recorded video inference

All three published demo runs use the saved `fireguard_s_local_v1/final/best.pt` checkpoint.

| Video | Inference size | Confidence | Deduplication IoU | Containment threshold |
|---|---:|---:|---:|---:|
| demo_3 | 768 | 0.25 | 0.30 | 0.90 |
| demo_4 | 768 | 0.25 | 0.30 | 0.90 |
| demo_5 | 960 | 0.25 | 0.30 | 0.90 |

Inference size 960 is a later inference setting, not the training resolution. For rectangular video, letterboxing scales the image according to the requested size and model stride; it does not stretch every frame into a 960-by-960 square.

The post-processing suppresses overlapping or contained detections of the same class while retaining detections from different classes. Its effect on object recall needs labeled data; cleaner overlays alone do not establish improved detection quality.

The demo_5 overlay displays processing FPS over a rolling window of 30 frame intervals. It includes decoding, inference, drawing, and writes to the encoder input, but excludes the final encoder drain. It is not a model-only GPU latency benchmark.

See [before/after videos and run settings](docs/videos/tests/README.md). Published videos are 720p viewing copies; their upload resolution does not indicate the model's inference size. The videos have no complete frame-level ground truth and do not establish accuracy or comparative superiority over YOLOv8.

## 6. Reproduction entry point

Use [train_local.py](train_local.py) and [TRAINING_LOCAL.md](TRAINING_LOCAL.md):

```powershell
.\.venv\Scripts\python.exe train_local.py --check
.\.venv\Scripts\python.exe -u train_local.py --phase prepare
.\.venv\Scripts\python.exe -u train_local.py --phase all
```

The preparation phase includes data checks, a smoke test, and baseline timing. The full phase runs the configured workflow and can reuse or resume existing work.

The pipeline originally writes final artifacts under:

```text
outputs/training/checkpoints/fireguard_s_local_v1/final/
```

The model currently used by the video runs is the copied checkpoint under:

```text
checkpoints/fireguard_s_local_v1/final/
```

The historical preparation review predates the completed server training. Use the saved stage logs and final model card for completed-epoch information.

## 7. Next steps

These tasks are proposed and are not claimed as completed:

1. **Preserve evaluation provenance.** Keep dataset manifests, split identities, checkpoint hashes, package versions, hardware details, and full evaluation reports together.
2. **Run an independent comparison.** Evaluate FireGuard and YOLOv8 standard on the same labeled holdout, with class mapping, image size, thresholds, and post-processing documented. Check overlap with training data before calling a dataset independent.
3. **Measure the resolution trade-off.** Compare 640, 768, and 960 on the same frames and hardware, reporting per-class precision, recall, mAP, model latency, and total processing FPS.
4. **Validate deduplication.** Compare predictions before and after suppression against labeled small flames and overlapping smoke regions.
5. **Evaluate long videos.** Measure missed incidents, detection delay, and false alarms over negative footage before selecting temporal alert rules.
6. **Validate deployment exports.** A saved ONNX artifact exists, but export availability alone does not establish output equivalence or deployment speed. Check numerical agreement and benchmark the intended runtime before deployment.

No new training, dataset evaluation, or deployment benchmark was run while writing this document.
