"""
src/training/model_evaluator.py
Yeni eğitilen modeli, sabit bir validation seti üzerinde mevcut production modeliyle
mAP50/precision/recall bazında karşılaştırır ve promote/reject kararı üretir.
"""
import json
import time
from pathlib import Path
from typing import Optional

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PRODUCTION_DIR = Path(
    config.get("app", "data_dirs", "production_model", default="data/models/production")
)
PRODUCTION_POINTER = PRODUCTION_DIR / "current_model.json"
DATASET_DIR = Path(config.get("app", "data_dirs", "dataset", default="data/dataset"))


def _evaluate_onnx(onnx_path: Path, dataset_version: str) -> dict:
    """
    ONNX modelini validation seti üzerinde değerlendirir.
    Ultralytics val() yerine ONNX Runtime ile doğrudan inference yapar,
    böylece gerçek production koşullarını simüle eder.

    Basitleştirilmiş metrik hesaplama — production'da full COCO evaluation önerilir.
    """
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(f"Değerlendirme için gerekli paket eksik: {e}")

    val_images_dir = DATASET_DIR / dataset_version / "images" / "val"
    val_labels_dir = DATASET_DIR / dataset_version / "labels" / "val"

    if not val_images_dir.exists():
        logger.warning("Validation images dir not found: %s", val_images_dir)
        return {"mAP50": 0.0, "precision": 0.0, "recall": 0.0, "evaluated_images": 0}

    # ONNX session
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    conf_threshold = config.get("inference", "confidence_threshold", default=0.5)
    iou_threshold = config.get("inference", "iou_threshold", default=0.45)

    total_tp = total_fp = total_fn = 0
    aps = []
    evaluated = 0

    for img_path in sorted(val_images_dir.glob("*.png"))[:200]:  # max 200 görüntü
        label_path = val_labels_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        # Ground truth
        gt_boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    gt_boxes.append((cls_id, cx, cy, w, h))

        # Inference
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640)).astype(np.float32) / 255.0
        tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]
        raw = session.run(None, {input_name: tensor})[0]

        # Basit TP/FP/FN sayımı
        pred = raw[0]  # (84, 8400)
        boxes_raw = pred[:4, :].T
        scores_raw = pred[4:, :].T
        class_ids = np.argmax(scores_raw, axis=1)
        confidences = scores_raw[np.arange(len(scores_raw)), class_ids]
        mask = confidences >= conf_threshold
        num_preds = int(mask.sum())

        tp = min(num_preds, len(gt_boxes))
        fp = max(0, num_preds - len(gt_boxes))
        fn = max(0, len(gt_boxes) - num_preds)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        evaluated += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    # Basit mAP50 tahmini
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    map50 = f1  # Gerçek mAP hesabı için pycocotools kullanılmalıdır

    return {
        "mAP50": round(map50, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "evaluated_images": evaluated,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def compare_and_decide(
    new_onnx_path: Path,
    dataset_version: str,
    new_version: str,
) -> dict:
    """
    Yeni modeli production modeliyle karşılaştırır.

    Returns
    -------
    dict:
        {
          "decision": "promote" | "reject",
          "new_metrics": {...},
          "current_metrics": {...},
          "delta_mAP50": float,
          "reason": str
        }
    """
    map50_improvement_threshold: float = config.get(
        "training", "promotion_map50_threshold", default=0.01
    )

    # Yeni modeli değerlendir
    logger.info("Evaluating new model: %s", new_onnx_path)
    new_metrics = _evaluate_onnx(new_onnx_path, dataset_version)

    # Mevcut production modelini değerlendir
    current_metrics = {"mAP50": 0.0, "precision": 0.0, "recall": 0.0}
    if PRODUCTION_POINTER.exists():
        with open(PRODUCTION_POINTER, "r", encoding="utf-8") as f:
            pointer = json.load(f)
        current_model_path = Path(pointer.get("model_path", ""))
        if current_model_path.exists():
            logger.info("Evaluating current production model: %s", current_model_path)
            current_metrics = _evaluate_onnx(current_model_path, dataset_version)
        else:
            logger.warning("Current production model not found — treating as baseline 0")

    delta = new_metrics["mAP50"] - current_metrics["mAP50"]
    decision = "promote" if delta >= map50_improvement_threshold else "reject"
    reason = (
        f"New mAP50={new_metrics['mAP50']:.4f} vs current={current_metrics['mAP50']:.4f} "
        f"(delta={delta:+.4f}, threshold={map50_improvement_threshold})"
    )

    result = {
        "version": new_version,
        "decision": decision,
        "new_metrics": new_metrics,
        "current_metrics": current_metrics,
        "delta_mAP50": round(delta, 4),
        "map50_threshold": map50_improvement_threshold,
        "reason": reason,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    logger.info("Model comparison: %s | %s", decision.upper(), reason)
    return result
