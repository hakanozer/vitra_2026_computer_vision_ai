"""
src/training/train_yolo.py
HOST üzerinde çalışan YOLOv8 eğitim scripti.
Apple Silicon M4 için device="mps" kullanır.

ÖNEMLİ: Bu script Docker konteyneri içinde ÇALIŞMAZ.
         Docker Desktop for Mac, konteynerlere Metal/MPS passthrough sağlamaz.
         HOST Python ortamında (venv içinde) çalıştırılır.

Çalıştırma:
    python -m src.training.train_yolo --dataset latest --epochs 50
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

from src.training.dataset_splitter import prepare_dataset
from src.training.test_evaluator import evaluate_on_test_split
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_DIR = Path(config.get("app", "data_dirs", "dataset", default="data/dataset"))
RUNS_DIR = Path(config.get("training", "runs_dir", default="data/training_runs"))


def create_dataset_yaml(dataset_version: str) -> Path:
    """
    Geriye donuk uyumluluk icin korundu.
    Yeni pipeline'da dataset dogrulama + split + data.yaml uretimi prepare_dataset ile yapilir.
    """
    yaml_path, _ = prepare_dataset(dataset_version)
    return yaml_path


def _extract_best_row_metrics(best_row: dict, headers) -> dict:
    map50_col = next((h for h in headers if "mAP50(B)" in h and "50-95" not in h), None)
    map5095_col = next((h for h in headers if "mAP50-95" in h), None)
    prec_col = next((h for h in headers if "precision" in h.lower()), None)
    recall_col = next((h for h in headers if "recall" in h.lower()), None)

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    return {
        "mAP50": _to_float(best_row.get(map50_col, 0.0)) if map50_col else 0.0,
        "mAP50_95": _to_float(best_row.get(map5095_col, 0.0)) if map5095_col else 0.0,
        "precision": _to_float(best_row.get(prec_col, 0.0)) if prec_col else 0.0,
        "recall": _to_float(best_row.get(recall_col, 0.0)) if recall_col else 0.0,
    }


def _select_best_checkpoint_by_map5095(run_dir: Path) -> tuple[Path, dict, dict | None]:
    """
    results.csv'deki val mAP50-95(B) kolonuna gore en iyi epoch checkpoint'ini best.pt yapar.
    checkpoint dosyasi bulunamazsa mevcut best.pt korunur.
    """
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    results_csv = run_dir / "results.csv"

    info = {
        "best_strategy": "ultralytics_default",
        "best_epoch": None,
        "best_map50_95": None,
    }

    if not results_csv.exists() or not best_pt.exists():
        return best_pt, info, None

    with open(results_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return best_pt, info, None

    headers = rows[0].keys()
    map_col = next((h for h in headers if "mAP50-95" in h), None)
    epoch_col = next((h for h in headers if h.strip().lower() == "epoch"), None)
    if map_col is None or epoch_col is None:
        return best_pt, info, None

    best_row = None
    best_score = float("-inf")
    for row in rows:
        try:
            score = float(row.get(map_col, 0.0))
        except Exception:
            score = 0.0
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return best_pt, info, None

    try:
        best_epoch = int(float(best_row.get(epoch_col, 0)))
    except Exception:
        best_epoch = 0

    epoch_files = {}
    for path in weights_dir.glob("epoch*.pt"):
        match = re.search(r"epoch(\d+)\.pt$", path.name)
        if not match:
            continue
        epoch_files[int(match.group(1))] = path

    selected = epoch_files.get(best_epoch) or epoch_files.get(best_epoch + 1)
    best_row_metrics = _extract_best_row_metrics(best_row, headers)

    if selected is not None:
        # best.pt'yi ezmeyelim; secilen checkpoint'i dogrudan kullanalim.
        chosen_model = selected
        info = {
            "best_strategy": "val_map50_95",
            "best_epoch": best_epoch,
            "best_map50_95": best_score,
            "selected_checkpoint": str(selected),
        }
    else:
        logger.warning(
            "Epoch checkpoint file not found for epoch=%d. Keeping Ultralytics best.pt",
            best_epoch,
        )
        info = {
            "best_strategy": "ultralytics_default",
            "best_epoch": best_epoch,
            "best_map50_95": best_score,
        }
        chosen_model = best_pt

    return chosen_model, info, best_row_metrics


def train(
    dataset_version: str = "latest",
    epochs: int = None,
    imgsz: int = None,
    batch: int = None,
    model_arch: str = None,
    device: str = "mps",
) -> dict:
    """
    YOLO eğitimini başlatır.

    Parameters
    ----------
    dataset_version : data/dataset/ altındaki versiyon klasörü
    epochs          : Eğitim epoch sayısı (config'den okunur, override edilebilir)
    imgsz           : Giriş görüntü boyutu
    batch           : Batch boyutu
    model_arch      : "yolov8n", "yolov8s", "yolov8m", vs.
    device          : "mps" (Apple Silicon M4) | "cpu" | "cuda:0" (Linux/NVIDIA)

    Returns
    -------
    dict : {"best_pt": str, "run_dir": str, "metrics": {...}}
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "ultralytics paketi bulunamadı. "
            "Host ortamında: pip install -r requirements-training.txt"
        )

    # Config'den varsayılanları oku
    epochs = epochs or config.get("training", "epochs", default=50)
    imgsz = imgsz or config.get("training", "imgsz", default=640)
    batch = batch or config.get("training", "batch", default=8)
    model_arch = model_arch or config.get("training", "model_arch", default="yolov8n")
    project = str(RUNS_DIR)
    run_name = f"train_{dataset_version}_{time.strftime('%Y%m%d_%H%M%S')}"

    yaml_path, split_report = prepare_dataset(dataset_version)

    # Küçük veri setleri için warmup (ısınma fazı) ve patience (erken durdurma) devre dışı bırakılır.
    # Aksi takdirde, model weights warmup iterasyon limiti altında kalıp güncellenemez.
    ds_path = DATASET_DIR / dataset_version
    train_labels_dir = ds_path / "train" / "labels"
    num_train_samples = len(list(train_labels_dir.glob("*.txt"))) if train_labels_dir.exists() else 0

    extra_train_args = {}
    if num_train_samples < 10:
        logger.info(
            "Small dataset detected (%d samples). Disabling warmup and early stopping for optimal learning.",
            num_train_samples
        )
        extra_train_args["warmup_epochs"] = 0.0
        extra_train_args["patience"] = 0
    else:
        extra_train_args["patience"] = config.get("training", "patience", default=20)

    logger.info(
        "Starting YOLO training: arch=%s epochs=%d imgsz=%d batch=%d device=%s extra_args=%s",
        model_arch,
        epochs,
        imgsz,
        batch,
        device,
        extra_train_args,
    )

    model = YOLO(f"{model_arch}.pt")  # Pretrained ağırlıkları indir

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        exist_ok=True,
        optimizer=config.get("training", "optimizer", default="AdamW"),
        lr0=config.get("training", "lr0", default=0.001),
        augment=True,
        save=True,
        save_period=1,
        val=True,
        plots=True,
        verbose=True,
        **extra_train_args
    )

    run_dir = Path(project) / run_name

    # En iyi modeli val mAP50-95'e gore sec
    best_pt, best_model_info, selected_metrics = _select_best_checkpoint_by_map5095(run_dir)

    # Metrikler
    metrics = {}
    if selected_metrics is not None and best_model_info.get("best_strategy") == "val_map50_95":
        metrics = selected_metrics
    elif hasattr(results, "results_dict"):
        rd = results.results_dict
        metrics = {
            "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision": float(rd.get("metrics/precision(B)", 0)),
            "recall": float(rd.get("metrics/recall(B)", 0)),
        }

    logger.info("Training complete. best.pt: %s metrics: %s", best_pt, metrics)

    # Egitim sonrasi test split degerlendirmesi
    test_report = evaluate_on_test_split(
        model_path=best_pt,
        data_yaml=yaml_path,
        device=device,
        imgsz=imgsz,
        batch=batch,
        run_name=run_name,
        run_dir=run_dir,
    )

    return {
        "best_pt": str(best_pt),
        "run_dir": str(run_dir),
        "metrics": metrics,
        "test_metrics": test_report.get("metrics", {}),
        "results_dir": test_report.get("results_dir"),
        "split_report": split_report,
        "best_model_info": best_model_info,
        "dataset_version": dataset_version,
        "run_name": run_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Vitra YOLO Eğitim Scripti (Host/MPS)")
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
