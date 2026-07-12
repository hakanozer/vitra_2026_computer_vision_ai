"""
src/dataset/labeling_queue.py
Etiketleme kuyruğunu yöneten sınıf — operatör onayı/reddi ve bbox kaydetme.
Etiketler COCO-benzeri JSON formatında data/labeling_queue/ altında saklanır.
"""
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABELING_QUEUE_DIR = Path(
    config.get("app", "data_dirs", "labeling_queue", default="data/labeling_queue")
)
DATASET_DIR = Path(
    config.get("app", "data_dirs", "dataset", default="data/dataset")
)


class LabelingQueueManager:
    """
    Operatör etiketleme iş akışını yönetir:
    - Bekleyen adayları listeler
    - Tek bir aday için detay döner (görüntü yolu + meta)
    - Bbox etiketini JSON olarak kaydeder
    - Onaylanan aday → data/dataset/unlabeled/ veya data/dataset/labeled/
    - Reddedilen aday → kuyruktan silinir
    """

    def list_pending(self, limit: int = 50) -> list[dict]:
        """Henüz etiketlenmemiş (status=pending) adayların listesini döner."""
        results = []
        for meta_file in sorted(LABELING_QUEUE_DIR.glob("*.json"))[:limit]:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") == "pending":
                results.append(
                    {
                        "id": meta["id"],
                        "camera_id": meta.get("camera_id", "unknown"),
                        "timestamp": meta.get("timestamp"),
                        "source": meta.get("source", "capture"),
                        "quality": meta.get("quality", {}),
                        "image_url": f"/labeling/image/{meta['id']}",
                    }
                )
        return results

    def get_candidate(self, sample_id: str) -> Optional[dict]:
        """Belirli bir adayın meta verisini döner."""
        meta_path = LABELING_QUEUE_DIR / f"{sample_id}.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_image_path(self, sample_id: str) -> Optional[Path]:
        """Adayın görüntü dosyasının yolunu döner."""
        img_path = LABELING_QUEUE_DIR / f"{sample_id}.png"
        return img_path if img_path.exists() else None

    def save_label(
        self,
        sample_id: str,
        annotations: list[dict],
        class_names: list[str],
        operator_id: str = "anonymous",
    ) -> bool:
        """
        Bbox etiketlerini COCO-benzeri JSON formatında kaydeder.

        Parameters
        ----------
        annotations : list of {"class_id": int, "bbox_xywh": [x, y, w, h], "confidence": float}
        class_names : ["defect_scratch", "defect_crack", …]
        operator_id : Etiketleyen operatörün kimliği (audit trail)
        """
        meta_path = LABELING_QUEUE_DIR / f"{sample_id}.json"
        if not meta_path.exists():
            logger.warning("Sample not found in queue: %s", sample_id)
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        label_data = {
            "id": sample_id,
            "image_file": f"{sample_id}.png",
            "labeled_at": time.time(),
            "operator_id": operator_id,
            "class_names": class_names,
            "annotations": annotations,
        }

        # Etiket dosyasını kaydet
        label_path = LABELING_QUEUE_DIR / f"{sample_id}_labels.json"
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)

        # Meta'yı güncelle
        meta["status"] = "labeled"
        meta["labeled_at"] = time.time()
        meta["operator_id"] = operator_id
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            "Label saved: id=%s operator=%s annotations=%d",
            sample_id,
            operator_id,
            len(annotations),
        )
        return True

    def approve_candidate(self, sample_id: str, dataset_version: str = "latest") -> bool:
        """
        Etiketlenmiş adayı onaylar ve data/dataset/{version}/ altına taşır.
        YOLO formatına dönüştürme bu adımda yapılır.
        """
        meta_path = LABELING_QUEUE_DIR / f"{sample_id}.json"
        label_path = LABELING_QUEUE_DIR / f"{sample_id}_labels.json"
        img_path = LABELING_QUEUE_DIR / f"{sample_id}.png"

        if not all(p.exists() for p in [meta_path, label_path, img_path]):
            logger.warning("Cannot approve %s: missing files", sample_id)
            return False

        # Dataset dizinini hazırla
        ds_dir = DATASET_DIR / dataset_version
        images_dir = ds_dir / "images" / "train"
        labels_dir = ds_dir / "labels" / "train"
        for d in [images_dir, labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Görüntüyü kopyala
        shutil.copy2(img_path, images_dir / f"{sample_id}.png")

        # Etiketleri YOLO .txt formatına dönüştür
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        img_h, img_w = img.shape[:2]

        yolo_lines = []
        for ann in label_data.get("annotations", []):
            cls_id = ann["class_id"]
            x, y, w, h = ann["bbox_xywh"]
            # YOLO format: class_id cx cy w h (normalized 0-1)
            cx_n = (x + w / 2) / img_w
            cy_n = (y + h / 2) / img_h
            w_n = w / img_w
            h_n = h / img_h
            yolo_lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        txt_path = labels_dir / f"{sample_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # Meta güncelle
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "approved"
        meta["dataset_version"] = dataset_version
        meta["approved_at"] = time.time()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            "Candidate approved: id=%s → dataset/%s (annotations=%d)",
            sample_id,
            dataset_version,
            len(label_data.get("annotations", [])),
        )
        return True

    def reject_candidate(self, sample_id: str, reason: str = "") -> bool:
        """Adayı reddeder — meta güncellenir, dosyalar kuyruğan kaldırılır."""
        meta_path = LABELING_QUEUE_DIR / f"{sample_id}.json"
        if not meta_path.exists():
            return False
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "rejected"
        meta["rejected_at"] = time.time()
        meta["rejection_reason"] = reason
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info("Candidate rejected: id=%s reason=%s", sample_id, reason)
        return True

    def get_queue_stats(self) -> dict:
        """Kuyruktaki aday sayılarını duruma göre özetler."""
        stats = {"pending": 0, "labeled": 0, "approved": 0, "rejected": 0, "total": 0}
        for meta_file in LABELING_QUEUE_DIR.glob("*.json"):
            if "_labels" in meta_file.name:
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                status = meta.get("status", "pending")
                stats[status] = stats.get(status, 0) + 1
                stats["total"] += 1
            except (json.JSONDecodeError, KeyError):
                pass
        return stats
