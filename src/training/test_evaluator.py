"""
src/training/test_evaluator.py
Egitim sonrasi test split'i uzerinde otomatik degerlendirme ve raporlama.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_ROOT = Path(config.get("training", "results_dir", default="results"))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_confusion_matrix(val_result) -> list[list[float]]:
    matrix = []
    confusion = getattr(val_result, "confusion_matrix", None)
    if confusion is None:
        return matrix

    raw = getattr(confusion, "matrix", None)
    if raw is None:
        return matrix

    try:
        return [[float(v) for v in row] for row in raw.tolist()]
    except Exception:
        return matrix


def save_test_report(
    *,
    run_name: str,
    run_dir: Path,
    metrics: dict,
    confusion_matrix: list[list[float]],
) -> dict:
    out_dir = RESULTS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) JSON raporu
    json_path = out_dir / "test_report.json"
    report_payload = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    # 2) CSV metrik raporu
    csv_path = out_dir / "test_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in ["precision", "recall", "f1_score", "mAP50", "mAP50_95"]:
            writer.writerow([key, metrics.get(key, 0.0)])

    # 3) Confusion Matrix CSV
    cm_path = out_dir / "confusion_matrix.csv"
    with open(cm_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if confusion_matrix:
            width = len(confusion_matrix[0])
            writer.writerow(["row/col"] + [f"c{idx}" for idx in range(width)])
            for row_idx, row in enumerate(confusion_matrix):
                writer.writerow([f"r{row_idx}"] + row)
        else:
            writer.writerow(["info", "not_available"])

    # 4) Egitim epoch metrikleri (varsa)
    train_history_src = run_dir / "results.csv"
    train_history_dst = out_dir / "training_history.csv"
    if train_history_src.exists():
        shutil.copy2(train_history_src, train_history_dst)

    logger.info("Test reports saved: %s", out_dir)
    return {
        "results_dir": str(out_dir),
        "test_report_json": str(json_path),
        "test_metrics_csv": str(csv_path),
        "confusion_matrix_csv": str(cm_path),
        "training_history_csv": str(train_history_dst) if train_history_src.exists() else None,
    }


def evaluate_on_test_split(
    *,
    model_path: Path,
    data_yaml: Path,
    device: str,
    imgsz: int,
    batch: int,
    run_name: str,
    run_dir: Path,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics paketi bulunamadı. Host ortamında: pip install -r requirements-training.txt"
        ) from exc

    model = YOLO(str(model_path))
    test_result = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        save_json=False,
        plots=True,
        verbose=True,
    )

    rd = getattr(test_result, "results_dict", {}) or {}
    precision = _safe_float(rd.get("metrics/precision(B)", 0.0))
    recall = _safe_float(rd.get("metrics/recall(B)", 0.0))
    map50 = _safe_float(rd.get("metrics/mAP50(B)", 0.0))
    map50_95 = _safe_float(rd.get("metrics/mAP50-95(B)", 0.0))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    confusion_matrix = _extract_confusion_matrix(test_result)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP50": map50,
        "mAP50_95": map50_95,
    }

    saved = save_test_report(
        run_name=run_name,
        run_dir=run_dir,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        **saved,
    }
