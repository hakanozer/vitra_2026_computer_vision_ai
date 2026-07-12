"""
src/pipeline/camera_reader.py
USB/IP kameradan frame okuyan, hatalara dayanıklı okuyucu.
Bağlantı koptuğunda otomatik yeniden bağlanır (watchdog pattern).
"""
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CameraReader:
    """
    Tek bir kamera kaynağını arka planda okuyan thread-safe sınıf.

    Attributes
    ----------
    source : int | str
        Kamera indeksi (0, 1, …) veya RTSP/HTTP URL.
    """

    def __init__(self, source: int | str = 0, name: str = "camera-0"):
        self.source = source
        self.name = name
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_delay: float = config.get(
            "app", "camera", "reconnect_delay_seconds", default=3.0
        )
        self._max_reconnect_attempts: int = config.get(
            "app", "camera", "max_reconnect_attempts", default=10
        )

    # ------------------------------------------------------------------ #
    # Yaşam döngüsü
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Kamera okuma thread'ini başlatır."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"capture-{self.name}", daemon=True
        )
        self._thread.start()
        logger.info("[%s] Capture thread started (source=%s)", self.name, self.source)

    def stop(self) -> None:
        """Thread'i durdurur ve kaynağı serbest bırakır."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info("[%s] CameraReader stopped", self.name)

    # ------------------------------------------------------------------ #
    # Frame erişimi
    # ------------------------------------------------------------------ #

    def get_frame(self) -> Optional[np.ndarray]:
        """Son başarıyla okunan frame'i döner; henüz frame yoksa None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_alive(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # İç döngü
    # ------------------------------------------------------------------ #

    def _open_capture(self) -> bool:
        """VideoCapture nesnesini açmayı dener. Başarıysa True döner."""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.source)
        if self._cap.isOpened():
            width = config.get("app", "camera", "width", default=1280)
            height = config.get("app", "camera", "height", default=720)
            fps = config.get("app", "camera", "fps", default=30)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            logger.info("[%s] Camera opened: %dx%d @%dfps", self.name, width, height, fps)
            return True
        logger.warning("[%s] Failed to open camera source: %s", self.name, self.source)
        return False

    def _capture_loop(self) -> None:
        """
        Ana okuma döngüsü.
        Bağlantı koparsa _max_reconnect_attempts kadar yeniden dener,
        ardından pes eder ve _running'i False yapar.
        """
        reconnect_count = 0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if reconnect_count >= self._max_reconnect_attempts:
                    logger.error(
                        "[%s] Max reconnect attempts reached (%d). Stopping.",
                        self.name,
                        self._max_reconnect_attempts,
                    )
                    self._running = False
                    break
                success = self._open_capture()
                if not success:
                    reconnect_count += 1
                    logger.warning(
                        "[%s] Reconnect attempt %d/%d in %.1fs …",
                        self.name,
                        reconnect_count,
                        self._max_reconnect_attempts,
                        self._reconnect_delay,
                    )
                    time.sleep(self._reconnect_delay)
                    continue
                reconnect_count = 0  # Başarıyla bağlandı, sayacı sıfırla

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("[%s] Frame read failed — will attempt reconnect", self.name)
                self._cap.release()
                continue

            with self._lock:
                self._frame = frame
