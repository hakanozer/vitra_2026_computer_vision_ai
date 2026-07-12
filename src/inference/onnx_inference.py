"""
src/inference/onnx_inference.py
ONNX Runtime tabanlı YOLOv8 inference — CPU/CoreML, arm64 uyumlu.

Not: Gerçek üretim ortamında (NVIDIA Linux sunucu) CUDAExecutionProvider
     veya TensorRTExecutionProvider kullanılır. Bu projede Mac M4 üzerinde
     CPUExecutionProvider / CoreMLExecutionProvider ile fonksiyonel eşdeğer
     çalışma sağlanır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# YOLOv8 varsayılan giriş boyutu
INPUT_SIZE = (640, 640)


@dataclass
class Detection:
    """Tek bir nesne tespiti."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]   # [x1, y1, x2, y2] — orijinal görüntü koordinatları
    bbox_xywh: list[float]   # [x, y, w, h]      — orijinal görüntü koordinatları


@dataclass
class InferenceResult:
    """Bir frame için inference sonuçları."""

    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_index: int = 0
    camera_id: str = ""
    has_low_confidence: bool = False   # Aktif öğrenme geri beslemesi için


class OnnxInference:
    """
    YOLOv8 ONNX modeliyle nesne tespiti yapar.

    Adım adım inference akışı:
    1. Preprocess: BGR → RGB, resize → 640×640, normalize → [0, 1], NCHW
    2. Run: session.run() ile raw output alınır
    3. Postprocess: raw output → bbox decode → NMS → Detection listesi
    """

    def __init__(self, session: ort.InferenceSession, class_names: list[str]):
        self._session = session
        self._class_names = class_names
        self._conf_threshold: float = config.get(
            "inference", "confidence_threshold", default=0.5
        )
        self._iou_threshold: float = config.get(
            "inference", "iou_threshold", default=0.45
        )
        self._low_conf_threshold: float = config.get(
            "inference", "low_confidence_feedback_threshold", default=0.35
        )
        # Model giriş adı ve boyutu
        self._input_name = session.get_inputs()[0].name
        logger.debug("ONNX input name: %s", self._input_name)

    def predict(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        camera_id: str = "",
    ) -> InferenceResult:
        """
        BGR frame alır, InferenceResult döner.
        """
        import time

        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocess
        input_tensor = self._preprocess(frame)

        # 2. Inference
        t0 = time.monotonic()
        raw_output = self._session.run(None, {self._input_name: input_tensor})
        inference_ms = (time.monotonic() - t0) * 1000

        # 3. Postprocess
        detections = self._postprocess(raw_output[0], orig_w, orig_h)

        # Aktif öğrenme: herhangi bir tespit düşük confidence'a sahip mi?
        has_low_conf = any(
            d.confidence < self._conf_threshold and d.confidence >= self._low_conf_threshold
            for d in detections
        )

        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_ms,
            frame_index=frame_index,
            camera_id=camera_id,
            has_low_confidence=has_low_conf,
        )

    # ------------------------------------------------------------------ #
    # Preprocess
    # ------------------------------------------------------------------ #

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR görüntüyü model giriş tensörüne dönüştürür.

        Adımlar:
        1. BGR → RGB: OpenCV varsayılan olarak BGR okur, model RGB bekler
        2. Resize: 640×640 — letterbox yerine basit resize (production'da letterbox önerilir)
        3. Normalize: uint8 [0,255] → float32 [0.0, 1.0]
        4. HWC → NCHW: (640,640,3) → (1,3,640,640)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]  # (1,3,H,W)
        return nchw

    # ------------------------------------------------------------------ #
    # Postprocess
    # ------------------------------------------------------------------ #

    def _postprocess(
        self, output: np.ndarray, orig_w: int, orig_h: int
    ) -> list[Detection]:
        """
        YOLOv8 ONNX çıktısını (1, 84, 8400) → Detection listesine dönüştürür.

        YOLOv8 ONNX çıktı formatı: (batch, 4+num_classes, num_anchors)
        - output[0, 0:4, :]  = cx, cy, w, h (normalized 0–1, 640×640'a göre)
        - output[0, 4:,  :]  = her sınıf için ham skor

        Scaling: 640×640 koordinatlarını orijinal görüntü boyutuna çevir.
        NMS: Üst üste binen kutuları temizle (cv2.dnn.NMSBoxes).
        """
        predictions = output[0]  # (84, 8400) — batch boyutu çıkarıldı

        # cx, cy, w, h + sınıf skorları
        boxes_raw = predictions[:4, :].T      # (8400, 4)
        scores_raw = predictions[4:, :].T     # (8400, num_classes)

        class_ids = np.argmax(scores_raw, axis=1)
        confidences = scores_raw[np.arange(len(scores_raw)), class_ids]

        # Eşik filtresi
        mask = confidences >= self._low_conf_threshold
        boxes_raw = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_raw) == 0:
            return []

        # cx,cy,w,h → x1,y1,w,h (640×640 koordinatları)
        scale_x = orig_w / INPUT_SIZE[0]
        scale_y = orig_h / INPUT_SIZE[1]

        x1 = (boxes_raw[:, 0] - boxes_raw[:, 2] / 2) * scale_x
        y1 = (boxes_raw[:, 1] - boxes_raw[:, 3] / 2) * scale_y
        w = boxes_raw[:, 2] * scale_x
        h = boxes_raw[:, 3] * scale_y

        # cv2.dnn.NMSBoxes için liste dönüşümü
        boxes_list = [[float(x), float(y), float(ww), float(hh)]
                      for x, y, ww, hh in zip(x1, y1, w, h)]
        confs_list = confidences.tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes_list, confs_list, self._conf_threshold, self._iou_threshold
        )
        if len(indices) == 0:
            return []

        detections = []
        for idx in indices.flatten():
            bx, by, bw, bh = boxes_list[idx]
            conf = confs_list[idx]
            cid = int(class_ids[idx])
            name = self._class_names[cid] if cid < len(self._class_names) else str(cid)
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=conf,
                    bbox_xyxy=[bx, by, bx + bw, by + bh],
                    bbox_xywh=[bx, by, bw, bh],
                )
            )
        return detections
