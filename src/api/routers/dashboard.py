"""
src/api/routers/dashboard.py
Dashboard veri endpoint'leri — gerçek zamanlı istatistikler, son tespitler.
"""
import time
import collections
from fastapi import APIRouter

router = APIRouter()

# Bellek içi son analizler (production'da Redis/TimescaleDB önerilir)
_recent_detections: collections.deque = collections.deque(maxlen=100)
_stats: dict = {
    "total_frames_processed": 0,
    "total_defects_detected": 0,
    "uptime_start": time.time(),
}

# En son frame durumu (canlı status takibi için)
_latest_frame_status: dict = {
    "status": "idle",       # idle | ok | defect
    "timestamp": 0.0,
    "detections": [],
}
_latest_frame_status_by_camera: dict[str, dict] = {}


def record_detection(camera_id: str, detections: list, inference_ms: float) -> None:
    """Inference sonuçlarını dashboard için kaydeder."""
    _stats["total_frames_processed"] += 1
    
    # Sadece sınıf adı "defect" ile başlayanları gerçek hata sayacına ekle
    defects = [d for d in detections if d.class_name.startswith("defect")]
    if defects:
        _stats["total_defects_detected"] += len(defects)
        
    has_defect = len(defects) > 0
    has_ok = any(d.class_name == "ok" for d in detections)
    frame_status = "defect" if has_defect else ("ok" if has_ok else "ok_implicit")

    # Her analiz edilen frame'i tabloda tut.
    _recent_detections.append({
        "camera_id": camera_id,
        "timestamp": time.time(),
        "inference_ms": inference_ms,
        "status": frame_status,
        "detections": [
            {"class_name": d.class_name, "confidence": d.confidence}
            for d in detections
        ],
    })

    # En son frame durumunu güncelle
    
    # Sınıf tespiti yoksa implicit_ok kabul edilir.
    status_payload = {
        "status": frame_status,
        "timestamp": time.time(),
        "detections": [
            {"class_name": d.class_name, "confidence": d.confidence}
            for d in detections
        ],
        "inference_ms": inference_ms,
        "camera_id": camera_id,
    }
    _latest_frame_status.update(status_payload)
    _latest_frame_status_by_camera[camera_id] = status_payload


@router.get("/stats")
def get_stats(camera_id: str | None = None):
    """Genel sistem istatistikleri."""
    from src.api.main import (
        ensure_inference_model_loaded,
        get_camera_model_assignment,
        model_loader,
        stream_manager,
    )
    if not model_loader.is_loaded:
        ensure_inference_model_loaded()
    uptime_s = time.time() - _stats["uptime_start"]
    selected_camera_id = camera_id or next(iter(stream_manager.list_camera_ids()), None)
    latest_frame_status = (
        _latest_frame_status_by_camera.get(selected_camera_id, _latest_frame_status)
        if selected_camera_id is not None
        else _latest_frame_status
    )
    return {
        "uptime_seconds": round(uptime_s, 1),
        "total_frames_processed": _stats["total_frames_processed"],
        "total_defects_detected": _stats["total_defects_detected"],
        "model_loaded": model_loader.is_loaded,
        "model_version": model_loader.model_metadata.get("version", "unknown"),
        "camera_stats": stream_manager.get_queue_stats(),
        "camera_ids": stream_manager.list_camera_ids(),
        "selected_camera_id": selected_camera_id,
        "selected_camera_model": get_camera_model_assignment(selected_camera_id) if selected_camera_id else None,
        "latest_frame_status": latest_frame_status,
        "timestamp": time.time(),
    }


@router.get("/recent-detections")
def recent_detections(limit: int = 20, camera_id: str | None = None):
    """Son analiz kayıtlarını döner."""
    items = list(_recent_detections)
    if camera_id:
        items = [item for item in items if item.get("camera_id") == camera_id]
    items = items[-limit:]
    return {"detections": list(reversed(items))}


@router.get("/labeling-summary")
def labeling_summary():
    """Etiketleme kuyruğu özetini döner."""
    from src.dataset.labeling_queue import LabelingQueueManager
    return LabelingQueueManager().get_queue_stats()
