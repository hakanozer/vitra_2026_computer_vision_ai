"""
src/api/routers/training.py
Eğitim tetikleme ve durum sorgulama endpoint'leri.
NOT: Eğitim HOST üzerinde çalışır; bu endpoint'ler subprocess başlatır veya
     eğitim durumunu raporlar.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Eğitim durumu (bellek içi — production'da Redis/DB kullanın)
_training_status: dict = {
    "running": False,
    "last_run": None,
    "last_result": None,
    "error": None,
}
_training_lock = threading.Lock()


class TrainRequest(BaseModel):
    dataset_version: str = "latest"
    epochs: Optional[int] = None
    imgsz: Optional[int] = None
    batch: Optional[int] = None
    arch: Optional[str] = None
    device: str = "mps"
    auto_promote: bool = False


def _run_training_job(req: TrainRequest) -> None:
    """Background'da çalışan eğitim görevı."""
    global _training_status
    with _training_lock:
        _training_status["running"] = True
        _training_status["error"] = None
        _training_status["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        cmd = [
            sys.executable, "-m", "src.training.train_yolo",
            "--dataset", req.dataset_version,
            "--device", req.device,
        ]
        if req.epochs:
            cmd += ["--epochs", str(req.epochs)]
        if req.imgsz:
            cmd += ["--imgsz", str(req.imgsz)]
        if req.batch:
            cmd += ["--batch", str(req.batch)]
        if req.arch:
            cmd += ["--arch", req.arch]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            train_result = json.loads(result.stdout)
            with _training_lock:
                _training_status["last_result"] = train_result

            if req.auto_promote:
                _auto_register_and_promote(train_result, req.dataset_version)
        else:
            with _training_lock:
                _training_status["error"] = result.stderr[-2000:]
    except Exception as exc:
        with _training_lock:
            _training_status["error"] = str(exc)
    finally:
        with _training_lock:
            _training_status["running"] = False


def _auto_register_and_promote(train_result: dict, dataset_version: str) -> None:
    """Eğitim bittikten sonra modeli registry'ye kaydedip promote eder."""
    from src.registry.model_registry import ModelRegistry
    from src.registry.model_promoter import ModelPromoter
    from src.training.export_onnx import export
    from src.training.model_evaluator import compare_and_decide

    pt_path = Path(train_result["best_pt"])
    onnx_path = pt_path.parent / "best.onnx"

    # Export
    export(pt_path, onnx_path)

    # Değerlendir
    metrics = train_result.get("metrics", {})
    registry = ModelRegistry()
    version = registry.register(pt_path, onnx_path, metrics, dataset_version)

    # Karşılaştır
    decision = compare_and_decide(onnx_path, dataset_version, version)
    if decision["decision"] == "promote":
        ModelPromoter().promote(version)


@router.post("/start")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Eğitimi background görevi olarak başlatır."""
    with _training_lock:
        if _training_status["running"]:
            raise HTTPException(status_code=409, detail="Training already running")

    background_tasks.add_task(_run_training_job, req)
    return {"status": "started", "dataset_version": req.dataset_version}


@router.get("/status")
def training_status():
    """Eğitim durumunu döner."""
    with _training_lock:
        return dict(_training_status)
