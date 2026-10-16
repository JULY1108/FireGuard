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

This project adapts training and inference code from
[ParsaVictor/FireGuuard_Fire_detection_smoke](https://github.com/ParsaVictor/FireGuuard_Fire_detection_smoke),
with local training fixes, model fine-tuning, detection deduplication, and video inference changes.

The original copyright and [MIT license](LICENSE) are retained. Dataset licenses and terms apply separately.
