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
        self._candidate_threshold: float = min(self._conf_threshold, self._low_conf_threshold)
        if self._low_conf_threshold > self._conf_threshold:
            logger.warning(
                "low_confidence_feedback_threshold (%.3f) is higher than confidence_threshold (%.3f). "
                "Using %.3f for candidate prefilter.",
                self._low_conf_threshold,
                self._conf_threshold,
                self._candidate_threshold,
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
        detections, has_low_conf = self._postprocess(raw_output[0], orig_w, orig_h)


        result = InferenceResult(
            detections=detections,
            inference_time_ms=inference_ms,
            frame_index=frame_index,
            camera_id=camera_id,
            has_low_confidence=has_low_conf,
        )
        # print result
        return result

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
    ) -> tuple[list[Detection], bool]:
        """
        YOLOv8 ONNX çıktısını (1, 84, 8400) → Detection listesine dönüştürür.

        YOLOv8 ONNX çıktı formatı: (batch, 4+num_classes, num_anchors)
        - output[0, 0:4, :]  = cx, cy, w, h (normalized 0–1, 640×640'a göre)
        - output[0, 4:,  :]  = her sınıf için ham skor

        Scaling: 640×640 koordinatlarını orijinal görüntü boyutuna çevir.
        NMS: Üst üste binen kutuları temizle (cv2.dnn.NMSBoxes).
        """
        if output.ndim == 3:
            predictions = output[0]
        elif output.ndim == 2:
            predictions = output
        else:
            logger.warning("Unexpected ONNX output shape: %s", output.shape)
            return [], False

        # Bazı export'larda (attrs, N), bazılarında (N, attrs) döner.
        # Burada her iki düzeni de destekleyip (N, attrs) üretiriz.
        expected_attrs = 4 + len(self._class_names)
        if predictions.shape[0] == expected_attrs and predictions.shape[1] != expected_attrs:
            pred = predictions.T
        elif predictions.shape[1] == expected_attrs:
            pred = predictions
        elif predictions.shape[0] < predictions.shape[1]:
            pred = predictions.T
        else:
            pred = predictions

        if pred.ndim != 2 or pred.shape[1] < 5:
            logger.warning("Unexpected prediction matrix shape after normalize: %s", pred.shape)
            return [], False

        attrs = pred.shape[1]
        num_classes = len(self._class_names)

        def _sigmoid(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x))

        class_ids: np.ndarray
        confidences: np.ndarray
        boxes_raw: np.ndarray
        use_xyxy_input = False

        # Format A: NMS export benzeri [x1, y1, x2, y2, conf, class_id]
        if attrs == 6:
            class_col = pred[:, 5]
            near_integer_class = np.all(np.isfinite(class_col)) and np.all(
                np.abs(class_col - np.round(class_col)) < 1e-4
            )
            if near_integer_class:
                use_xyxy_input = True
                boxes_raw = pred[:, :4].astype(np.float32)
                confidences = pred[:, 4].astype(np.float32)
                class_ids = np.clip(np.round(class_col), 0, max(num_classes - 1, 0)).astype(int)
            else:
                boxes_raw = pred[:, :4]
                scores_raw = pred[:, 4:]
                if scores_raw.size == 0:
                    logger.warning("No class scores in ONNX output with shape: %s", predictions.shape)
                    return [], False
                if np.min(scores_raw) < 0.0 or np.max(scores_raw) > 1.0:
                    scores_raw = _sigmoid(scores_raw)
                class_ids = np.argmax(scores_raw, axis=1)
                confidences = scores_raw[np.arange(len(scores_raw)), class_ids]

        # Format B: YOLOv5 tarzı [cx, cy, w, h, obj, cls1..clsN]
        elif attrs >= 5 + num_classes and num_classes > 0:
            boxes_raw = pred[:, :4]
            obj = pred[:, 4:5]
            cls_scores = pred[:, 5:5 + num_classes]
            if np.min(obj) < 0.0 or np.max(obj) > 1.0:
                obj = _sigmoid(obj)
            if np.min(cls_scores) < 0.0 or np.max(cls_scores) > 1.0:
                cls_scores = _sigmoid(cls_scores)
            fused_scores = obj * cls_scores
            class_ids = np.argmax(fused_scores, axis=1)
            confidences = fused_scores[np.arange(len(fused_scores)), class_ids]

        # Format C: YOLOv8 tarzı [cx, cy, w, h, cls1..clsN]
        else:
            boxes_raw = pred[:, :4]
            scores_raw = pred[:, 4:]
            if scores_raw.size == 0:
                logger.warning("No class scores in ONNX output with shape: %s", predictions.shape)
                return [], False
            if np.min(scores_raw) < 0.0 or np.max(scores_raw) > 1.0:
                scores_raw = _sigmoid(scores_raw)
            class_ids = np.argmax(scores_raw, axis=1)
            confidences = scores_raw[np.arange(len(scores_raw)), class_ids]

        # Final conf eşiğinin altında ama low_conf eşiğinin üstünde kalanlar,
        # aktif öğrenme için sinyal olarak tutulur.
        has_low_conf_candidate = bool(
            np.any((confidences >= self._low_conf_threshold) & (confidences < self._conf_threshold))
        )

        if confidences.size > 0:
            logger.debug(
                "Postprocess stats: shape=%s attrs=%d conf_max=%.4f conf_mean=%.4f",
                pred.shape,
                attrs,
                float(np.max(confidences)),
                float(np.mean(confidences)),
            )

        # Aday filtresi: confidence_threshold düşürüldüğünde gerçekten daha fazla aday geçsin.
        mask = confidences >= self._candidate_threshold
        boxes_raw = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        logger.debug("Postprocess candidates after threshold %.3f: %d", self._candidate_threshold, len(boxes_raw))

        if len(boxes_raw) == 0:
            return [], has_low_conf_candidate

        scale_x = orig_w / INPUT_SIZE[0]
        scale_y = orig_h / INPUT_SIZE[1]

        if use_xyxy_input:
            x1_raw = boxes_raw[:, 0]
            y1_raw = boxes_raw[:, 1]
            x2_raw = boxes_raw[:, 2]
            y2_raw = boxes_raw[:, 3]

            # Bazı export'larda xyxy normalize [0,1] gelebilir.
            if np.max(np.abs(boxes_raw)) <= 2.0:
                x1 = x1_raw * orig_w
                y1 = y1_raw * orig_h
                x2 = x2_raw * orig_w
                y2 = y2_raw * orig_h
            else:
                x1 = x1_raw
                y1 = y1_raw
                x2 = x2_raw
                y2 = y2_raw

            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
        else:
            # cx,cy,w,h → x1,y1,w,h (640×640 koordinatları)
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
            logger.debug("Postprocess NMS kept 0 boxes (conf=%.3f iou=%.3f)", self._conf_threshold, self._iou_threshold)
            return [], has_low_conf_candidate

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
        logger.debug("Postprocess final detections: %d", len(detections))
        return detections, has_low_conf_candidate
