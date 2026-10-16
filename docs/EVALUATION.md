# Evaluation Results

Run: **fireguard_s_local_v1**  
Checkpoint: [best.pt](../checkpoints/fireguard_s_local_v1/final/best.pt)

## 1. Test-set object detection

The post-training evaluation cell calls:

```python
mt = model.val(
    data=YAML, split="test", imgsz=640,
    conf=0.001, iou=0.6, plots=True,
)
```

The local runner retains this test-set evaluation and copies its metrics into the final model card. Therefore, the saved detection metrics below are reported as the **test-set results from the training pipeline**, rather than the training validation curve.

The preparation audit recorded 3,532 images in the merged FASDD_CV + D-Fire test split. This is the pipeline's constructed holdout; it is not the untouched official D-Fire test split or a separate external benchmark.

| Metric | Test-set result |
|---|---:|
| mAP@0.50 | **68.39%** |
| mAP@0.50:0.95 | **41.83%** |
| Fire AP@0.50 | **69.50%** |
| Smoke AP@0.50 | **67.28%** |
| Precision | **73.66%** |
| Recall | **58.96%** |

Values are transcribed from the final model card. The evaluation split is established by tracing the evaluation cell and model-card generation; the metrics block itself does not repeat a split field. The raw evaluator output and dataset manifest are not included here, and this documentation update did not rerun evaluation.

The low conf=0.001 allows collection of predictions for detection evaluation and precision-recall curves. Precision/recall reported by the evaluator must not be described as measurements at the later video confidence of 0.25.

## 2. Validation image-trigger results

The threshold-calibration step explicitly uses **validation**, with 2,490 negative images and 1,087 positive images, totaling 3,577.

| Confidence threshold | Negative-image trigger rate | Positive-image trigger rate | Evidence |
|---|---:|---:|---|
| 0.10 | 0.763% | 97.884% | Saved threshold_calibration.json and model_card.json |
| 0.25 | **0.482%** | **96.044%** | Previously recorded result supplied in the user's screenshot |

The saved calibration recommends 0.10 for its target of no more than 1% negative-image triggers. The published demo videos instead use 0.25.

The screenshot's rounded 0.25 results are consistent with 12/2,490 negative images triggering and 1,044/1,087 positive images triggering. Those counts are inferred from the reported percentages and denominators, not newly recovered per-image predictions.

A trigger is **any detection in an image**, without matching the predicted class or bounding box against ground truth. Consequently:

- Positive-image trigger rate is not bounding-box recall.
- Negative-image trigger rate is not false alarms per camera per day.
- These validation figures are not independent test-set mAP results.

## 3. Comparison scope

The earlier screenshot also compared the 0.25 validation trigger results against figures quoted from the upstream author's README. Because the author's data split and measurement protocol were not established as identical, that table is not a controlled comparison.

No claim that FireGuard outperforms YOLOv8 standard is made from these results. A two-model comparison needs both checkpoints evaluated on the same labeled split with the same protocol.

## Evidence

- [Post-training evaluation cell](../FireGuard_Pipeline.ipynb): split="test".
- [Local runner](../train_local.py): retains test-set detection evaluation and changes threshold calibration to validation.
- Local artifacts: final/model_card.json and threshold_calibration.json.
- The user's screenshot: historical 0.25 validation trigger rates.
- [Training record](../PLAN_V2.md): dataset preparation and completed training.
