"""
src/api/routers/camera.py
Kamera yönetimi endpoint'leri.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class AddCameraRequest(BaseModel):
    camera_id: str
    source: int | str
    queue_size: int = 30


class CaptureNowRequest(BaseModel):
    camera_id: str | None = None       # Boş bırakılırsa varsayılan kamera kullanılır
    source: str = "manual_capture"     # Meta veride görünecek kaynak etiketi
    force_add: bool = True             # True: kalite filtresini atlar, her zaman kaydeder


@router.get("/list")
def list_cameras():
    """Kayıtlı kameraları ve kuyruk istatistiklerini döner."""
    from src.api.main import stream_manager
    return stream_manager.get_queue_stats()


@router.post("/add")
def add_camera(req: AddCameraRequest):
    """Yeni bir kamera ekler."""
    from src.api.main import stream_manager
    stream_manager.add_camera(req.camera_id, req.source, req.queue_size)
    return {"status": "added", "camera_id": req.camera_id}


@router.delete("/{camera_id}")
def remove_camera(camera_id: str):
    """Kamerayı kaldırır."""
    from src.api.main import stream_manager
    stream_manager.remove_camera(camera_id)
    return {"status": "removed", "camera_id": camera_id}


@router.post("/capture-now")
def capture_now(req: CaptureNowRequest):
    """
    Kameranın o anki görüntüsünü ANINDA yakalar ve etiketleme kuyruğuna ekler.
    30 saniyelik otomatik döngüyü beklemeden, kameraya bir nesne tutup
    manuel olarak veri toplamak için kullanılır.
    """
    from src.api.main import candidate_manager, stream_manager
    from src.utils.config_loader import config

    camera_id = req.camera_id or config.get(
        "app", "pipeline", "default_camera_id", default="camera-0"
    )

    frame = stream_manager.get_latest_frame(camera_id)
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"'{camera_id}' için henüz okunmuş bir frame yok. "
                "Kamera bağlantısını ve source ayarını kontrol edin."
            ),
        )

    sample_id = candidate_manager.process_frame(
        frame, camera_id, source=req.source, force_add=req.force_add
    )

    if sample_id is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Görüntü kalite filtresinden geçemedi (bulanık/karanlık/düşük kontrast). "
                "force_add=true göndererek filtreyi atlayabilirsiniz."
            ),
        )

    return {
        "status": "captured",
        "sample_id": sample_id,
        "camera_id": camera_id,
        "image_url": f"/api/labeling/image/{sample_id}",
    }