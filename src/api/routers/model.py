"""
src/api/routers/model.py
Model registry, karşılaştırma, promote ve rollback endpoint'leri.
"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class PromoteRequest(BaseModel):
    version: str


class EvaluateRequest(BaseModel):
    onnx_path: str
    dataset_version: str = "latest"
    new_version: str = "pending"


class CameraModelBindingRequest(BaseModel):
    model_version: str = "__production__"


@router.get("/registry")
def list_registry():
    """Registry'deki tüm model versiyonlarını listeler."""
    from src.registry.model_registry import ModelRegistry
    return {"versions": ModelRegistry().list_versions()}


@router.get("/production")
def current_production():
    """Mevcut production model bilgisini döner."""
    from src.registry.model_promoter import ModelPromoter
    info = ModelPromoter().get_current_production()
    if info is None:
        raise HTTPException(status_code=404, detail="No production model set")
    return info


@router.post("/promote")
def promote_model(req: PromoteRequest):
    """Belirtilen versiyonu production'a geçirir."""
    from src.registry.model_promoter import ModelPromoter
    success = ModelPromoter().promote(req.version)
    if not success:
        raise HTTPException(status_code=400, detail=f"Promote failed for version: {req.version}")
    return {"status": "promoted", "version": req.version}


@router.post("/rollback")
def rollback_model():
    """Bir önceki production modeline geri döner."""
    from src.registry.model_promoter import ModelPromoter
    success = ModelPromoter().rollback()
    if not success:
        raise HTTPException(status_code=400, detail="No previous model to roll back to")
    return {"status": "rolled_back"}


@router.post("/evaluate")
def evaluate_model(req: EvaluateRequest):
    """Yeni model ile mevcut production modelini karşılaştırır."""
    from src.training.model_evaluator import compare_and_decide
    onnx_path = Path(req.onnx_path)
    if not onnx_path.exists():
        raise HTTPException(status_code=404, detail=f"ONNX file not found: {req.onnx_path}")
    result = compare_and_decide(onnx_path, req.dataset_version, req.new_version)
    return result


@router.post("/register")
def register_model(
    pt_path: str,
    onnx_path: str,
    dataset_version: str,
    map50: float = 0.0,
    precision: float = 0.0,
    recall: float = 0.0,
):
    """Modeli registry'ye kaydeder."""
    from src.registry.model_registry import ModelRegistry
    metrics = {"mAP50": map50, "precision": precision, "recall": recall}
    version = ModelRegistry().register(
        Path(pt_path), Path(onnx_path), metrics, dataset_version
    )
    return {"status": "registered", "version": version}


@router.get("/camera-bindings")
def list_camera_bindings():
    """Kameralara atanmış model bağlarını döner."""
    from src.api.main import list_camera_model_assignments, stream_manager

    return {
        "bindings": list_camera_model_assignments(),
        "camera_ids": stream_manager.list_camera_ids(),
        "production_binding": "__production__",
    }


@router.put("/camera-bindings/{camera_id}")
def set_camera_binding(camera_id: str, req: CameraModelBindingRequest):
    """Belirli bir kameraya registry modeli veya production modeli atar."""
    from src.api.main import set_camera_model_binding, stream_manager

    if camera_id not in stream_manager.list_camera_ids():
        raise HTTPException(status_code=404, detail=f"Camera not registered: {camera_id}")

    try:
        assignment = set_camera_model_binding(camera_id, req.model_version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "updated",
        "assignment": assignment,
    }
