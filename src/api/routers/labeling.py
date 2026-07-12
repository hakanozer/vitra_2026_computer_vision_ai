"""
src/api/routers/labeling.py
Operatör etiketleme arayüzü endpoint'leri.
"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class LabelRequest(BaseModel):
    annotations: list[dict]
    class_names: list[str]
    operator_id: str = "anonymous"


class ApproveRequest(BaseModel):
    dataset_version: str = "latest"


class RejectRequest(BaseModel):
    reason: str = ""


@router.get("/queue")
def list_pending(limit: int = 50):
    """Etiketlenmemiş aday listesi."""
    from src.api.main import candidate_manager
    from src.dataset.labeling_queue import LabelingQueueManager
    mgr = LabelingQueueManager()
    return {"candidates": mgr.list_pending(limit=limit)}


@router.get("/queue/stats")
def queue_stats():
    """Etiketleme kuyruğu istatistikleri."""
    from src.dataset.labeling_queue import LabelingQueueManager
    return LabelingQueueManager().get_queue_stats()


@router.get("/image/{sample_id}")
def get_image(sample_id: str):
    """Adaya ait görüntüyü döner."""
    from src.dataset.labeling_queue import LabelingQueueManager
    path = LabelingQueueManager().get_image_path(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/png")


@router.get("/candidate/{sample_id}")
def get_candidate(sample_id: str):
    """Adayın meta verisini döner."""
    from src.dataset.labeling_queue import LabelingQueueManager
    meta = LabelingQueueManager().get_candidate(sample_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return meta


@router.post("/label/{sample_id}")
def save_label(sample_id: str, req: LabelRequest):
    """Bbox etiketlerini kaydeder."""
    from src.dataset.labeling_queue import LabelingQueueManager
    success = LabelingQueueManager().save_label(
        sample_id, req.annotations, req.class_names, req.operator_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Sample not found in queue")
    return {"status": "labeled", "sample_id": sample_id}


@router.post("/approve/{sample_id}")
def approve_candidate(sample_id: str, req: ApproveRequest):
    """Etiketlenmiş adayı onaylayıp dataset'e taşır."""
    from src.dataset.labeling_queue import LabelingQueueManager
    success = LabelingQueueManager().approve_candidate(sample_id, req.dataset_version)
    if not success:
        raise HTTPException(status_code=400, detail="Approval failed. Check logs.")
    return {"status": "approved", "sample_id": sample_id, "dataset_version": req.dataset_version}


@router.post("/reject/{sample_id}")
def reject_candidate(sample_id: str, req: RejectRequest):
    """Adayı reddeder."""
    from src.dataset.labeling_queue import LabelingQueueManager
    LabelingQueueManager().reject_candidate(sample_id, req.reason)
    return {"status": "rejected", "sample_id": sample_id}
