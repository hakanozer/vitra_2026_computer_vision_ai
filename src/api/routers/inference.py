"""
src/api/routers/inference.py
Inference durum, hot-swap ve konfigürasyon endpoint'leri.
"""
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


@router.get("/status")
def inference_status():
    """Inference motorunun durumunu döner."""
    from src.api.main import ensure_inference_model_loaded, model_loader
    if not model_loader.is_loaded:
        ensure_inference_model_loaded()
    return {
        "model_loaded": model_loader.is_loaded,
        "model_metadata": model_loader.model_metadata,
        "timestamp": time.time(),
    }


@router.post("/reload-model")
def reload_model():
    """
    Hot-swap: Production pointer'ı yeniden okuyarak modeli değiştirir.
    Servis yeniden başlatılmadan çalışır.
    """
    from src.api.main import ensure_inference_model_loaded, model_loader

    success = ensure_inference_model_loaded(force_reload=True)
    if not success:
        raise HTTPException(status_code=500, detail="Model reload failed. Check logs.")

    return {
        "status": "reloaded",
        "model_metadata": model_loader.model_metadata,
        "timestamp": time.time(),
    }


class InferenceConfigRequest(BaseModel):
    confidence_threshold: float | None = None
    iou_threshold: float | None = None


@router.patch("/config")
def update_inference_config(req: InferenceConfigRequest):
    """Inference eşiklerini runtime'da günceller."""
    from src.utils.config_loader import config
    changes = {}
    if req.confidence_threshold is not None:
        config._data.setdefault("inference", {})["confidence_threshold"] = req.confidence_threshold
        changes["confidence_threshold"] = req.confidence_threshold
    if req.iou_threshold is not None:
        config._data.setdefault("inference", {})["iou_threshold"] = req.iou_threshold
        changes["iou_threshold"] = req.iou_threshold
    return {"updated": changes}
