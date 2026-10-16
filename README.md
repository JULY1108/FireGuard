# FireGuard

Fire and smoke detection using a fine-tuned YOLO26s model.

## Completed training

- Base weights: COCO-pretrained `yolo26s.pt`.
- Classes: `0 = fire`, `1 = smoke`.
- Data: FASDD_CV and D-Fire, with class alignment, label auditing, perceptual deduplication, and group-based splitting.
- Full-network fine-tuning: **72 epochs at 512**, then **48 epochs at 640**.
- Download trained weights: [best.pt](checkpoints/fireguard_s_local_v1/final/best.pt) (20.3 MB).
- The PyTorch checkpoint is included in this repository; the ONNX export remains stored separately.

See [training records, saved metrics, and next steps](PLAN_V2.md). Metrics are recorded results; their evaluation and calibration limits are explained there.

## Network architecture and fine-tuning

See [the detailed architecture and training guide](docs/TRAINING_ARCHITECTURE.md) for backbone/neck/head structure, multi-scale features, end-to-end prediction, the 512 → 640 schedule, optimizer settings, augmentation, and checkpoint verification.

## Evaluation results

### Test-set object detection

The post-training pipeline evaluates `best.pt` on the constructed FASDD_CV + D-Fire test split.
The preparation audit recorded **3,532 test images**. This is the pipeline's holdout, rather than the untouched official D-Fire test split or a separate external benchmark.

Evaluation settings: **imgsz=640, conf=0.001, iou=0.6, split=test**.

| Metric | Test-set result |
|---|---:|
| mAP@0.50 | **68.39%** |
| mAP@0.50:0.95 | **41.83%** |
| Fire AP@0.50 | **69.50%** |
| Smoke AP@0.50 | **67.28%** |
| Precision | **73.66%** |
| Recall | **58.96%** |

These values come from the saved final model card. The evaluation cell uses `split=test` before model-card generation.
The raw evaluator report and split manifest have not been rechecked here. Evaluator precision and recall are not measurements at the later video confidence of 0.25.

### Validation image-trigger results

The validation split contains **2,490 negative images** and **1,087 positive images**, totaling **3,577**.

| Confidence threshold | Negative-image trigger rate | Positive-image trigger rate |
|---|---:|---:|
| 0.10 | 0.763% | 97.884% |
| 0.25 | **0.482%** | **96.044%** |

A trigger means any detection in an image, without checking class or bounding-box IoU.
Positive-image trigger rate is not object recall; negative-image trigger rate is not false alarms per camera per day.

The saved calibration recommends **0.10** for its target of at most 1% negative-image triggers.
The published demo videos use **0.25**. The 0.25 row is a historical recorded result, not a newly recomputed measurement.

### Interpretation

These detection and trigger results use different metrics and data splits. They do not establish superiority over YOLOv8 standard or the upstream author's model; that requires evaluation on the same labeled split with the same protocol.

See [the detailed evaluation report](docs/EVALUATION.md). No new evaluation was run for this documentation update.

## Video examples

[Watch demo_3, demo_4, and demo_5 before and after detection](docs/videos/tests/README.md).

The recorded runs use confidence 0.25 and class-aware duplicate suppression. Demo_3 and demo_4 use inference size 768; demo_5 uses 960 and displays processing FPS. Published videos are 720p viewing copies.

These runs perform video detection and annotation. They do not send email alerts.

## Training entry point

Create the Python environment and install the dependencies described in [TRAINING_LOCAL.md](TRAINING_LOCAL.md). Dataset downloads require network access, disk space, and acceptance of the source datasets' terms.

```powershell
.\.venv\Scripts\python.exe train_local.py --check
.\.venv\Scripts\python.exe -u train_local.py --phase prepare
.\.venv\Scripts\python.exe -u train_local.py --phase all
```

The entry point executes and adapts cells from `FireGuard_Pipeline.ipynb`; that notebook remains a required dependency. `build_pipeline_notebook.py` is its source generator. The local runner applies corrections described in the training review.

Datasets, intermediate checkpoints, the Python environment, and generated training outputs are not included in the public source repository. The final best.pt checkpoint is included.

## Repository

| Path | Purpose |
|---|---|
| [train_local.py](train_local.py) | Local training orchestration and fixes |
| [FireGuard_Pipeline.ipynb](FireGuard_Pipeline.ipynb) | Training cells loaded by the runner |
| [build_pipeline_notebook.py](build_pipeline_notebook.py) | Training notebook generator |
| [fireguard_core.py](fireguard_core.py) | Detection, deduplication, overlays, and video processing |
| [test_detection_dedup.py](test_detection_dedup.py) | Existing deduplication checks |
| [compare_fire_smoke_models.py](compare_fire_smoke_models.py) | Historical model comparison; requires the adjacent YOLOv8 project and its data preparation tools |
| [TRAINING_LOCAL.md](TRAINING_LOCAL.md) | Local environment and training instructions |
| [TRAINING_REVIEW.md](TRAINING_REVIEW.md) | Historical preparation audit and corrections |
| [PLAN_V2.md](PLAN_V2.md) | Completed training record and next steps |
| [docs/videos/tests/](docs/videos/tests/README.md) | Local before/after video examples |

## Attribution and license

This project incorporates code from
[ParsaVictor/FireGuuard_Fire_detection_smoke](https://github.com/ParsaVictor/FireGuuard_Fire_detection_smoke).
Its original MIT copyright and permission notice are retained in [LICENSE](LICENSE).

**The retained MIT notice is not a blanket license for the trained weights or dependencies.**
Ultralytics states that its software and YOLO trained/fine-tuned models are covered by AGPL-3.0 by default,
with a separate Enterprise License option. This also concerns the included YOLO26-derived best.pt;
the upstream FireGuard MIT notice does not override those model terms.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the applicable sources and scope.
Datasets and source footage retain their respective rights and terms.
No Enterprise License is granted or claimed, and this clarification does not independently relicense the entire repository.
