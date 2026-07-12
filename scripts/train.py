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
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vitra YOLO Eğitim CLI")
    parser.add_argument("--dataset", default="latest", help="Dataset versiyon adı")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--arch", default=None, help="yolov8n | yolov8s | yolov8m ...")
    parser.add_argument(
        "--device",
        default="mps",
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

    if not ModelPromoter().promote(version):
        raise RuntimeError(f"Model production'a alınamadı: {version}")

    logger.info("Training output promoted to production: %s", version)
    result["onnx_path"] = str(onnx_path)
    result["model_version"] = version
    result["promoted"] = True

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
