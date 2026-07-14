"""
src/dataset/candidate_manager.py
Kalite kontrolünden geçen frame'leri ham veri havuzuna (raw_captures) kaydeder
ve etiketleme kuyruğuna (labeling_queue) taşır.
"""
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.quality.image_quality import ImageQualityAnalyzer, QualityReport
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_CAPTURES_DIR = Path(config.get("app", "data_dirs", "raw_captures", default="data/raw_captures"))
LABELING_QUEUE_DIR = Path(config.get("app", "data_dirs", "labeling_queue", default="data/labeling_queue"))


class CandidateManager:
    """
    Görüntü kalite filtresini geçen frame'leri:
    1. data/raw_captures/ altına PNG olarak kaydeder
    2. Meta veriyi (zaman damgası, kamera ID, kalite skoru) JSON olarak yazar
    3. Etiketleme kuyruğuna (labeling_queue) taşır

    Klasör yapısı:
        data/raw_captures/{YYYY-MM-DD}/{uuid}.png
        data/raw_captures/{YYYY-MM-DD}/{uuid}.json   ← meta
        data/labeling_queue/{uuid}.png               ← symlink veya kopya
        data/labeling_queue/{uuid}.json
    """

    def __init__(self):
        self._analyzer = ImageQualityAnalyzer()
        RAW_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        LABELING_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        source: str = "capture",
        force_add: bool = False,
    ) -> Optional[str]:
        """
        Frame'i analiz eder; kalite eşiğini geçiyorsa kaydeder.

        Parameters
        ----------
        frame      : BGR numpy array
        camera_id  : kamera tanımlayıcısı (meta veriye eklenir)
        source     : "capture" | "low_confidence_feedback" (aktif öğrenme geri beslemesi)
        force_add  : True ise kalite kontrolünü atlar (operatör manuel ekleme)

        Returns
        -------
        str  : Kaydedilen dosyanın UUID'si, reddedildiyse None
        """
        report: QualityReport = self._analyzer.analyze(frame)

        if not force_add and not report.is_acceptable:
            logger.debug(
                "Frame from '%s' rejected: %s", camera_id, report.rejection_reason
            )
            return None

        sample_id = str(uuid.uuid4())
        date_str = time.strftime("%Y-%m-%d")
        day_dir = RAW_CAPTURES_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        # PNG kaydet
        img_path = day_dir / f"{sample_id}.png"
        cv2.imwrite(str(img_path), frame)

        # Meta JSON kaydet
        meta = {
            "id": sample_id,
            "camera_id": camera_id,
            "timestamp": time.time(),
            "date": date_str,
            "source": source,
            "quality": {
                "blur_score": report.blur_score,
                "brightness": report.brightness,
                "contrast": report.contrast,
                "is_acceptable": report.is_acceptable,
            },
            "status": "pending",  # pending | labeled | approved | rejected
        }
        meta_path = day_dir / f"{sample_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Labeling queue'ya kopyala
        lq_img = LABELING_QUEUE_DIR / f"{sample_id}.png"
        lq_meta = LABELING_QUEUE_DIR / f"{sample_id}.json"
        shutil.copy2(img_path, lq_img)
        shutil.copy2(meta_path, lq_meta)

        logger.info(
            "Frame saved to dataset: id=%s camera=%s source=%s blur=%.1f",
            sample_id,
            camera_id,
            source,
            report.blur_score,
        )
        return sample_id

    # imageid ile imagei file olarak silen fonksiyon
    def delete_image(self, imageid: str) -> bool:
        deleted_any = False

        # labeling_queue düz dizindir
        queue_targets = [
            LABELING_QUEUE_DIR / f"{imageid}.png",
            LABELING_QUEUE_DIR / f"{imageid}.json",
            LABELING_QUEUE_DIR / f"{imageid}_labels.json",
        ]
        for target in queue_targets:
            if target.exists():
                target.unlink()
                deleted_any = True

        # raw_captures altında tarih klasörleri olduğu için recursive aranır
        for target in RAW_CAPTURES_DIR.rglob(f"{imageid}.png"):
            target.unlink()
            deleted_any = True
        for target in RAW_CAPTURES_DIR.rglob(f"{imageid}.json"):
            target.unlink()
            deleted_any = True

        return deleted_any
