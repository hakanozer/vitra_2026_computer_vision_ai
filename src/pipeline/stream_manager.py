"""
src/pipeline/stream_manager.py
Çoklu kamera kaynağını ve frame queue'larını tek noktadan yöneten orkestratör.
"""
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from src.pipeline.camera_reader import CameraReader
from src.pipeline.frame_queue import FramePacket, FrameQueue
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamManager:
    """
    Birden fazla CameraReader + FrameQueue çiftini yönetir.
    Her kamera için ayrı bir üretici thread çalışır;
    tüketici (inference worker) get_next_frame() ile sıradaki frame'i alır.
    """

    def __init__(self):
        self._cameras: Dict[str, CameraReader] = {}
        self._queues: Dict[str, FrameQueue] = {}
        self._producer_threads: Dict[str, threading.Thread] = {}
        self._running = False
        self._frame_counter: Dict[str, int] = {}
        self._latest_annotated_frames: Dict[str, np.ndarray] = {}
        self._annotated_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Kamera yönetimi
    # ------------------------------------------------------------------ #

    def add_camera(
        self,
        camera_id: str,
        source: int | str,
        queue_size: int = 30,
    ) -> None:
        if camera_id in self._cameras:
            logger.warning("Camera '%s' already registered.", camera_id)
            return
        reader = CameraReader(source=source, name=camera_id)
        self._cameras[camera_id] = reader
        self._queues[camera_id] = FrameQueue(maxsize=queue_size)
        self._frame_counter[camera_id] = 0
        logger.info("Camera '%s' registered (source=%s)", camera_id, source)

    def remove_camera(self, camera_id: str) -> None:
        if camera_id not in self._cameras:
            return
        self._cameras[camera_id].stop()
        del self._cameras[camera_id]
        del self._queues[camera_id]
        del self._frame_counter[camera_id]
        logger.info("Camera '%s' removed.", camera_id)

    # ------------------------------------------------------------------ #
    # Yaşam döngüsü
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Tüm kameraları ve üretici thread'leri başlatır."""
        self._running = True
        for cam_id, reader in self._cameras.items():
            reader.start()
            t = threading.Thread(
                target=self._producer_loop,
                args=(cam_id,),
                name=f"producer-{cam_id}",
                daemon=True,
            )
            self._producer_threads[cam_id] = t
            t.start()
        logger.info("StreamManager started with %d camera(s).", len(self._cameras))

    def stop(self) -> None:
        self._running = False
        for reader in self._cameras.values():
            reader.stop()
        for t in self._producer_threads.values():
            t.join(timeout=5)
        logger.info("StreamManager stopped.")

    # ------------------------------------------------------------------ #
    # Frame erişimi
    # ------------------------------------------------------------------ #

    def get_next_frame(
        self, camera_id: str, timeout: float = 1.0
    ) -> Optional[FramePacket]:
        q = self._queues.get(camera_id)
        if q is None:
            return None
        return q.get(timeout=timeout)

    def get_latest_frame(self, camera_id: str):
        """
        Kameranın en son okuduğu frame'i, kuyruktan TÜKETMEDEN döner.
        Manuel "şimdi yakala" gibi anlık istekler için kullanılır —
        inference kuyruğunu etkilemez.
        """
        reader = self._cameras.get(camera_id)
        if reader is None:
            return None
        return reader.get_frame()

    def set_latest_annotated_frame(self, camera_id: str, frame: np.ndarray) -> None:
        with self._annotated_lock:
            self._latest_annotated_frames[camera_id] = frame.copy()

    def get_latest_annotated_frame(self, camera_id: str) -> Optional[np.ndarray]:
        with self._annotated_lock:
            frame = self._latest_annotated_frames.get(camera_id)
            return frame.copy() if frame is not None else None

    def get_queue_stats(self) -> Dict[str, dict]:
        return {
            cam_id: {
                "qsize": q.qsize,
                "dropped": q.dropped_count,
                "camera_alive": self._cameras[cam_id].is_alive,
            }
            for cam_id, q in self._queues.items()
        }

    # ------------------------------------------------------------------ #
    # İç döngü
    # ------------------------------------------------------------------ #

    def _producer_loop(self, camera_id: str) -> None:
        """
        CameraReader'dan frame alıp FrameQueue'ya iten üretici döngüsü.
        Inference FPS'ini aşmamak için config'deki hedef FPS'e göre bekleme ekler.
        """
        target_fps: float = config.get("app", "pipeline", "target_fps", default=15.0)
        interval = 1.0 / target_fps

        reader = self._cameras[camera_id]
        q = self._queues[camera_id]

        while self._running:
            t0 = time.monotonic()
            frame = reader.get_frame()
            if frame is not None:
                self._frame_counter[camera_id] += 1
                packet = FramePacket(
                    frame=frame,
                    camera_id=camera_id,
                    frame_index=self._frame_counter[camera_id],
                )
                q.put(packet)
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)