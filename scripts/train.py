#!/usr/bin/env python3
"""
scripts/train.py
YOLO eğitimini HOST üzerinde başlatan CLI wrapper.
Çalıştırma: python scripts/train.py --dataset latest --epochs 50 --device mps
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Proje kökünü Python path'ine ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry.model_promoter import ModelPromoter
from src.registry.model_registry import ModelRegistry
from src.training.export_onnx import export
from src.training.train_yolo import train
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _f1_from_metrics(metrics: dict) -> float:
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    if (precision + recall) <= 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _promotion_gate(test_metrics: dict, val_metrics: dict, split_report: dict) -> tuple[bool, str, dict]:
    enabled = bool(config.get("training", "promotion_gate", "enabled", default=True))
    if not enabled:
        return True, "promotion gate disabled", {"metrics_source": "disabled", "used_metrics": {}}

    min_map5095 = float(config.get("training", "promotion_gate", "min_test_map50_95", default=0.05))
    min_f1 = float(config.get("training", "promotion_gate", "min_test_f1", default=0.05))
    min_precision = float(config.get("training", "promotion_gate", "min_test_precision", default=0.01))
    min_test_images = int(config.get("training", "promotion_gate", "min_test_images", default=5))
    tiny_test_policy = str(
        config.get("training", "promotion_gate", "insufficient_test_policy", default="use_val")
    ).strip().lower()

    counts = split_report.get("counts", {}) if isinstance(split_report, dict) else {}
    test_count = int(counts.get("test", 0) or 0)

    source = "test"
    source_metrics = {
        "mAP50_95": float(test_metrics.get("mAP50_95", 0.0)),
        "precision": float(test_metrics.get("precision", 0.0)),
        "f1_score": float(test_metrics.get("f1_score", _f1_from_metrics(test_metrics))),
    }

    if test_count < min_test_images:
        if tiny_test_policy == "block":
            reason = (
                f"test gate blocked: insufficient test samples ({test_count} < {min_test_images})"
            )
            return False, reason, {
                "metrics_source": "test",
                "used_metrics": source_metrics,
                "test_count": test_count,
            }
        if tiny_test_policy == "skip":
            reason = (
                f"test gate skipped: insufficient test samples ({test_count} < {min_test_images})"
            )
            return True, reason, {
                "metrics_source": "test",
                "used_metrics": source_metrics,
                "test_count": test_count,
            }
        if tiny_test_policy == "use_val":
            source = "val_fallback"
            source_metrics = {
                "mAP50_95": float(val_metrics.get("mAP50_95", 0.0)),
                "precision": float(val_metrics.get("precision", 0.0)),
                "f1_score": float(_f1_from_metrics(val_metrics)),
            }

    test_map5095 = source_metrics["mAP50_95"]
    test_f1 = source_metrics["f1_score"]
    test_precision = source_metrics["precision"]

    ok = (
        test_map5095 >= min_map5095
        and test_f1 >= min_f1
        and test_precision >= min_precision
    )
    reason = (
        f"test gate ({source}): map50_95={test_map5095:.4f}>={min_map5095:.4f}, "
        f"f1={test_f1:.4f}>={min_f1:.4f}, "
        f"precision={test_precision:.4f}>={min_precision:.4f}"
    )
    details = {
        "metrics_source": source,
        "used_metrics": source_metrics,
        "test_count": test_count,
        "min_test_images": min_test_images,
        "insufficient_test_policy": tiny_test_policy,
    }
    return ok, reason, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Vitra YOLO Eğitim CLI")
    parser.add_argument("--dataset", default="latest", help="Dataset versiyon adı")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--arch", default=None, help="yolov8n | yolov8s | yolov8m ...")
    parser.add_argument(
        "--device",
        default="mps", # windows için cuda:0, mac için mps, cpu için cpu
        help="mps (Mac M4) | cpu | cuda:0 (Linux/NVIDIA)",
    )
    args = parser.parse_args()

    result = train(
        dataset_version=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_arch=args.arch,
        device=args.device,
    )

    pt_path = Path(result["best_pt"])
    onnx_path = pt_path.parent / "best.onnx"
    export(pt_path, onnx_path)

    version = ModelRegistry().register(
        pt_path=pt_path,
        onnx_path=onnx_path,
        metrics=result.get("metrics", {}),
        dataset_version=result["dataset_version"],
    )

    gate_ok, gate_reason, gate_details = _promotion_gate(
        result.get("test_metrics", {}),
        result.get("metrics", {}),
        result.get("split_report", {}),
    )
    promoted = False
    if gate_ok:
        if not ModelPromoter().promote(version):
            raise RuntimeError(f"Model production'a alınamadı: {version}")
        promoted = True
        logger.info("Training output promoted to production: %s", version)
    else:
        logger.warning("Promotion skipped for version=%s (%s)", version, gate_reason)

    result["onnx_path"] = str(onnx_path)
    result["model_version"] = version
    result["promoted"] = promoted
    result["promotion_gate"] = {
        "passed": gate_ok,
        "reason": gate_reason,
        **gate_details,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
