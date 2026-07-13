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
import json
import time
from pathlib import Path

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_DIR = Path(config.get("app", "data_dirs", "dataset", default="data/dataset"))
RUNS_DIR = Path(config.get("training", "runs_dir", default="data/training_runs"))


def create_dataset_yaml(dataset_version: str) -> Path:
    """
    Ultralytics için data.yaml dosyası üretir.
    Bu dosya dataset konumunu ve sınıf isimlerini tanımlar.
    """
    ds_path = DATASET_DIR / dataset_version
    class_names: list[str] = config.get(
        "training", "class_names", default=["defect_scratch", "defect_crack", "defect_dent", "ok"]
    )

    has_val = (ds_path / "images" / "val").exists()
    val_split = "images/val" if has_val else "images/train"
    if not has_val:
        logger.warning("images/val bulunamadı, val olarak images/train kullanılıyor.")

    yaml_content = f"""# Otomatik üretildi: {time.strftime('%Y-%m-%dT%H:%M:%S')}
path: {ds_path.resolve()}
train: images/train
val: {val_split}

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = ds_path / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info("Dataset YAML created: %s", yaml_path)
    return yaml_path


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

    yaml_path = create_dataset_yaml(dataset_version)

    # Küçük veri setleri için warmup (ısınma fazı) ve patience (erken durdurma) devre dışı bırakılır.
    # Aksi takdirde, model weights warmup iterasyon limiti altında kalıp güncellenemez.
    ds_path = DATASET_DIR / dataset_version
    train_labels_dir = ds_path / "labels" / "train"
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
        plots=True,
        verbose=True,
        **extra_train_args
    )

    # En iyi model yolu
    best_pt = Path(project) / run_name / "weights" / "best.pt"

    # Metrikler
    metrics = {}
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        metrics = {
            "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision": float(rd.get("metrics/precision(B)", 0)),
            "recall": float(rd.get("metrics/recall(B)", 0)),
        }

    logger.info("Training complete. best.pt: %s metrics: %s", best_pt, metrics)

    return {
        "best_pt": str(best_pt),
        "run_dir": str(Path(project) / run_name),
        "metrics": metrics,
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
