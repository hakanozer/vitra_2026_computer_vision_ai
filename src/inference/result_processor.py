"""
src/inference/result_processor.py
Inference sonuçlarını işler:
  - IoT/alarm tetikleme
  - Düşük confidence frame'lerini labeling queue'ya geri besler (aktif öğrenme)
  - Dashboard'a yayınlar
"""
import time
from typing import Callable, Optional

import cv2
import numpy as np

from src.api.routers.dashboard import record_detection
from src.inference.onnx_inference import Detection, InferenceResult
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResultProcessor:
    """
    Inference sonuçlarını yorumlar ve ilgili eylemleri tetikler.

    Sorumluluklar:
    1. Tespit var mı? → alarm callback'i çağır
    2. Düşük confidence var mı? → geri besleme callback'i çağır
    3. Görselleştirme için frame'i annotate et
    """

    def __init__(
        self,
        alarm_callback: Optional[Callable[[InferenceResult], None]] = None,
        feedback_callback: Optional[Callable[[np.ndarray, str], None]] = None,
    ):
        """
        Parameters
        ----------
        alarm_callback    : Tespit bulunduğunda çağrılır (MQTT/Modbus alarm için)
        feedback_callback : Düşük confidence frame'i geri beslemek için çağrılır
                           Signature: (frame: np.ndarray, camera_id: str) -> None
        """
        self._alarm_callback = alarm_callback
        self._feedback_callback = feedback_callback
        self._class_names: list[str] = config.get(
            "inference", "class_names", default=["defect"]
        )

    def process(self, result: InferenceResult, frame: np.ndarray) -> np.ndarray:
        """
        Sonuçları işler ve annotate edilmiş frame döner.

        Aktif öğrenme geri beslemesi:
        Eğer herhangi bir tespit düşük confidence (<conf_threshold) ama
        low_conf_threshold üstündeyse, bu frame otomatik olarak labeling
        queue'ya geri yazılır. Bu sayede model belirsiz olduğu örnekleri
        öğrenmeye devam eder.
        """
        record_detection(
            camera_id=result.camera_id,
            detections=result.detections,
            inference_ms=result.inference_time_ms,
        )

        # 1. Alarm tetikle
        if result.detections and self._alarm_callback:
            self._alarm_callback(result)

        # 2. Aktif öğrenme geri beslemesi
        if result.has_low_confidence and self._feedback_callback:
            logger.debug(
                "[%s] Low confidence detections — feeding frame back to labeling queue",
                result.camera_id,
            )
            self._feedback_callback(frame.copy(), result.camera_id)

        # 3. Annotate
        annotated = self._draw_detections(frame.copy(), result.detections)
        return annotated

    def _draw_detections(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Tespit kutularını ve etiketleri frame üzerine çizer."""
        colors = {
            "defect_scratch": (0, 0, 255),   # Kırmızı
            "defect_crack": (0, 165, 255),    # Turuncu
            "defect_dent": (0, 255, 255),     # Sarı
            "ok": (0, 255, 0),                # Yeşil
        }
        default_color = (255, 0, 0)  # Mavi

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = colors.get(det.class_name, default_color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Class: {det.class_name} | Confidence: {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label_top = max(0, y1 - th - 6)
            y_label_bottom = max(0, y1)
            cv2.rectangle(frame, (x1, y_label_top), (x1 + tw + 6, y_label_bottom), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, max(12, y_label_bottom - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        return frame
