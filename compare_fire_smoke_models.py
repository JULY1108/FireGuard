"""Compare both projects' weights on one reproduced, balanced 200-image test set."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parent
OTHER = ROOT.parent / "2. yolov8-fire-smoke-detection-main"
OUT = ROOT / "model_comparison"
DATA = OUT / "flameye_test_200"
for name, folder in (("YOLO_CONFIG_DIR", "yolo_config"), ("MPLCONFIGDIR", "mpl_config")):
    path = OUT / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
sys.path.insert(0, str(OTHER))

import torch
import requests
import yaml
from ultralytics import YOLO


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_data(prep):
    manifest = DATA / "manifest.jsonl"
    summary_path = DATA / "dataset_summary.json"
    if manifest.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        items = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        assert len(items) == 200 and summary["category_counts"] == {f"test/{c}": 50 for c in prep.LARGE_TARGETS["test"]}
        assert all((DATA / item["image_path"]).is_file() and (DATA / item["label_path"]).is_file() for item in items)
        return items, summary

    DATA.mkdir(parents=True, exist_ok=True)
    selection_path = DATA / "selected_test_rows.jsonl"
    if selection_path.exists():
        selected = [json.loads(line) for line in selection_path.read_text(encoding="utf-8").splitlines() if line]
    else:
        session = requests.Session()
        session.headers.update({"User-Agent": "fire-smoke-model-comparison/1.0"})
        candidates = prep.collect_candidates(session, "test", prep.LARGE_TARGETS["test"], 100, 12000)
        selected = []
        for category, count in prep.LARGE_TARGETS["test"].items():
            for row in prep.choose_diverse(candidates[category], count):
                selected.append({"category": category, "row": row})
        assert len(selected) == 200
        selection_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

    (DATA / "images" / "test").mkdir(parents=True, exist_ok=True)
    (DATA / "labels" / "test").mkdir(parents=True, exist_ok=True)
    progress = DATA / "manifest.progress.jsonl"
    done = {}
    if progress.exists():
        done = {item["image_id"]: item for item in
                (json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line)}
    image_session = requests.Session()
    image_session.headers.update({"User-Agent": "fire-smoke-model-comparison/1.0"})
    with progress.open("a", encoding="utf-8") as log:
        for index, selected_row in enumerate(selected):
            category, row = selected_row["category"], selected_row["row"]
            image_id = prep.safe_name(str(row.get("image_id", f"sample_{index}")))
            filename = f"{index:03d}_{category}_{image_id}.jpg"
            image_rel, label_rel = f"images/test/{filename}", f"labels/test/{Path(filename).stem}.txt"
            image_path, label_path = DATA / image_rel, DATA / label_rel
            if row["image_id"] in done:
                continue
            image = prep.download_image(image_session, row["image"]["src"])
            image.save(image_path, format="JPEG", quality=92)
            annotations = row.get("annotations") or []
            prep.write_yolo_label(label_path, annotations)
            item = {"dataset": prep.DATASET_ID, "api_split": "test", "split": "test",
                    "category": category, "source": row.get("source", "unknown"),
                    "image_id": row["image_id"], "image_path": image_rel,
                    "label_path": label_rel, "width": image.width, "height": image.height,
                    "annotations": annotations}
            log.write(json.dumps(item, ensure_ascii=False) + "\n")
            log.flush()
            done[row["image_id"]] = item
            if (index + 1) % 25 == 0:
                print(f"Prepared {index+1}/200 test images", flush=True)

    items = [done[row["row"]["image_id"]] for row in selected]
    assert len(items) == 200
    progress.replace(manifest)
    categories = Counter(item["category"] for item in items)
    sources = Counter(item["source"] for item in items)
    summary = {"dataset": prep.DATASET_ID, "profile": "large test split",
               "class_mapping": prep.CLASS_TO_ID, "split_counts": {"test": len(items)},
               "category_counts": {f"test/{k}": v for k, v in categories.items()},
               "source_counts": dict(sources),
               "selection_note": "Rebuilt from the YOLOv8 project's dataset source and sampling code; original manifest was not preserved."}
    info = requests.get(f"https://huggingface.co/api/datasets/{prep.DATASET_ID}", timeout=30)
    info.raise_for_status()
    summary["dataset_revision"] = info.json().get("sha")
    summary["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary["image_set_sha256"] = hashlib.sha256("".join(
        f"{item['image_id']}:{hashlib.sha256((DATA/item['image_path']).read_bytes()).hexdigest()}\n"
        for item in items).encode()).hexdigest()
    (DATA / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(DATA), "train": "images/test", "val": "images/test", "test": "images/test",
         "names": {0: "smoke", 1: "fire"}}, sort_keys=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return items, summary


def main():
    assert torch.cuda.is_available(), "CUDA GPU required for a comparable inference run"
    assert (OTHER / "weights").is_dir()
    prep = load_module(OTHER / "prepare_experiment_dataset.py", "comparison_dataset_prep")
    evaluator = load_module(OTHER / "run_full_experiment.py", "comparison_metrics")
    items, data_summary = prepare_data(prep)
    weights = {
        "FireGuard YOLO26s": ROOT / "checkpoints/fireguard_s_local_v1/final/best.pt",
        "YOLOv8 baseline": OTHER / "weights/baseline_best.pt",
        "YOLOv8 standard": OTHER / "weights/standard_best.pt",
        "YOLOv8 strong augmentation": OTHER / "weights/strong_augmentation_best.pt",
    }
    assert all(path.is_file() for path in weights.values())
    rows = []
    for name, path in weights.items():
        print(f"Evaluating {name}: {path}", flush=True)
        model = YOLO(str(path))
        result, _ = evaluator.evaluate(model, items, DATA, confidence=.25, nms_iou=.30,
                                      imgsz=768, device="0")
        row = {"model": name, "weight": str(path), "class_names": model.names, **result}
        rows.append(row)
        del model
        torch.cuda.empty_cache()
        print(f"  P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f} "
              f"TP={result['tp']} FP={result['fp']} FN={result['fn']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["model", "confidence", "nms_iou", "evaluation_iou", "tp", "fp", "fn",
              "precision", "recall", "f1", "avg_inference_ms"]
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    historical = json.loads((OTHER / "results" / "summary.json").read_text(encoding="utf-8-sig"))
    reference = next(row for row in historical["best_configurations"]
                     if row["model"] == "baseline" and row["confidence"] == .25 and row["nms_iou"] == .30)
    current_baseline = next(row for row in rows if row["model"] == "YOLOv8 baseline")
    baseline_check = {key: {"stored": reference[key], "reproduced": current_baseline[key],
                            "matches": reference[key] == current_baseline[key]}
                      for key in ("tp", "fp", "fn", "precision", "recall", "f1")}
    result = {"dataset": data_summary, "historical_baseline_check": baseline_check,
              "evaluation": {"images": 200, "imgsz": 768,
              "confidence": .25, "nms_iou": .30, "box_iou_threshold": .50,
              "metric": "class-aware one-to-one box matching, same evaluator as YOLOv8 project"},
              "models": rows, "caveat": "Original 200-image manifest was absent; dataset was reconstructed using the same source and repository sampling code. Exact historical image identity cannot be confirmed."}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# FireGuard 与 YOLOv8 权重同集对比", "",
             "所有模型在同一重建的 200 张测试集上，以 imgsz=768、conf=0.25、NMS IoU=0.30 和匹配 IoU=0.50 评估。", "",
             "| 模型 | P | R | F1 | TP | FP | FN | 平均推理(ms/图) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
                     f"{row['tp']} | {row['fp']} | {row['fn']} | {row['avg_inference_ms']:.2f} |")
    lines += ["", f"测试集按 YOLOv8 项目的公开数据源和抽样代码重建（数据版本 {data_summary['dataset_revision']}，每类场景 50 张）。",
              f"对照原始结果记录，Baseline 的 TP/FP/FN 和 P/R/F1 均匹配：{baseline_check['tp']['matches']}。",
              "原始 manifest 未保留，无法核实 200 张图片的逐张身份。类别按模型 names 映射后比较；所有模型使用相同置信度、NMS IoU 和框匹配 IoU。"]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved comparison to {OUT}", flush=True)


if __name__ == "__main__":
    main()
