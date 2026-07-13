"""
src/api/routers/dashboard.py
Dashboard veri endpoint'leri — gerçek zamanlı istatistikler, son tespitler.
"""
import time
import collections
from fastapi import APIRouter

router = APIRouter()

# Bellek içi son tespitler (production'da Redis/TimescaleDB önerilir)
_recent_detections: collections.deque = collections.deque(maxlen=100)
_stats: dict = {
    "total_frames_processed": 0,
    "total_defects_detected": 0,
    "uptime_start": time.time(),
}


def record_detection(camera_id: str, detections: list, inference_ms: float) -> None:
    """Inference sonuçlarını dashboard için kaydeder."""
    _stats["total_frames_processed"] += 1
    print(f"Recording detection for camera_id={camera_id}, detections={detections}, inference_ms={inference_ms}")
    if detections:
        _stats["total_defects_detected"] += len(detections)
        _recent_detections.append({
            "camera_id": camera_id,
            "timestamp": time.time(),
            "inference_ms": inference_ms,
            "detections": [
                {"class_name": d.class_name, "confidence": d.confidence}
                for d in detections
            ],
        })


@router.get("/stats")
def get_stats():
    """Genel sistem istatistikleri."""
    from src.api.main import ensure_inference_model_loaded, model_loader, stream_manager
    if not model_loader.is_loaded:
        ensure_inference_model_loaded()
    uptime_s = time.time() - _stats["uptime_start"]
    return {
        "uptime_seconds": round(uptime_s, 1),
        "total_frames_processed": _stats["total_frames_processed"],
        "total_defects_detected": _stats["total_defects_detected"],
        "model_loaded": model_loader.is_loaded,
        "model_version": model_loader.model_metadata.get("version", "unknown"),
        "camera_stats": stream_manager.get_queue_stats(),
        "timestamp": time.time(),
    }


@router.get("/recent-detections")
def recent_detections(limit: int = 20):
    """Son tespit edilen defect'leri döner."""
    items = list(_recent_detections)[-limit:]
    print(f"Returning {items} recent detections (limit={limit})")
    return {"detections": list(reversed(items))}


@router.get("/labeling-summary")
def labeling_summary():
    """Etiketleme kuyruğu özetini döner."""
    from src.dataset.labeling_queue import LabelingQueueManager
    return LabelingQueueManager().get_queue_stats()
