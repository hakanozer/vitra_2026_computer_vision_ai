"""
src/pipeline/frame_queue.py
Producer-Consumer frame kuyruğu — kamera okuyucu ile inference arasındaki tampon.
"""
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FramePacket:
    """Kuyruktaki bir frame ve meta verisi."""

    frame: np.ndarray
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0


class FrameQueue:
    """
    Thread-safe, maxsize'lı bir frame kuyruğu.
    Kuyruk dolunca en eski frame atılır (drop-oldest policy),
    böylece inference daima en güncel frame'i görür.
    """

    def __init__(self, maxsize: int = 30):
        self._q: queue.Queue[FramePacket] = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._lock = threading.Lock()

    def put(self, packet: FramePacket) -> None:
        """Frame'i kuyruğa ekler; kuyruk doluysa en eskiyi atar."""
        if self._q.full():
            try:
                self._q.get_nowait()
                with self._lock:
                    self._dropped += 1
            except queue.Empty:
                pass
        self._q.put_nowait(packet)

    def get(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Kuyruktan bir frame alır; timeout süresi içinde frame gelmezse None döner."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    def clear(self) -> None:
        """Kuyruğu temizler."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
