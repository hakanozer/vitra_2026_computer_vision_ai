"""
src/api/main.py
FastAPI uygulaması — inference servisi, dashboard API ve etiketleme endpoint'leri.
"""
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import camera, dashboard, inference, labeling, model, training
from src.dataset.candidate_manager import CandidateManager
from src.inference.model_loader import ModelLoader
from src.inference.onnx_inference import OnnxInference
from src.inference.result_processor import ResultProcessor
from src.iot.modbus_client import ModbusAlarmWriter
from src.iot.mqtt_client import MqttPublisher
from src.pipeline.stream_manager import StreamManager
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Uygulama durumu (singleton nesneler)
# ------------------------------------------------------------------ #
stream_manager = StreamManager()
model_loader = ModelLoader()
candidate_manager = CandidateManager()
mqtt_publisher = MqttPublisher()
modbus_writer = ModbusAlarmWriter()

_inference_engine: Optional[OnnxInference] = None
_result_processor: Optional[ResultProcessor] = None
_inference_thread: Optional[threading.Thread] = None
_inference_running = False


# ------------------------------------------------------------------ #
# Alarm callback
# ------------------------------------------------------------------ #

def _alarm_callback(result) -> None:
    mqtt_publisher.publish_alarm(
        result.camera_id, result.detections, result.frame_index
    )
    if result.detections:
        modbus_writer.write_alarm(active=True)


def _feedback_callback(frame: np.ndarray, camera_id: str) -> None:
    """Düşük confidence frame'lerini labeling queue'ya geri besler."""
    candidate_manager.process_frame(
        frame, camera_id, source="low_confidence_feedback", force_add=True
    )


def ensure_inference_model_loaded(force_reload: bool = False) -> bool:
    """Production modelini yükler ve inference engine'i senkronize eder."""
    global _inference_engine, _result_processor

    if force_reload:
        model_ready = model_loader.reload()
    elif model_loader.is_loaded:
        model_ready = True
    else:
        model_ready = model_loader.load_production_model()

    if not model_ready:
        return False

    class_names: list[str] = config.get(
        "inference", "class_names", default=["defect_scratch", "defect_crack", "defect_dent", "ok"]
    )
    _inference_engine = OnnxInference(model_loader.session, class_names)
    if _result_processor is None:
        _result_processor = ResultProcessor(
            alarm_callback=_alarm_callback,
            feedback_callback=_feedback_callback,
        )
    return True


# ------------------------------------------------------------------ #
# Inference döngüsü
# ------------------------------------------------------------------ #

def _run_inference_loop() -> None:
    global _inference_running
    camera_id: str = config.get("app", "pipeline", "default_camera_id", default="camera-0")
    capture_interval: float = config.get("app", "pipeline", "capture_save_every_n_seconds", default=5.0)
    last_capture = 0.0

    while _inference_running:
        if not model_loader.is_loaded:
            time.sleep(1)
            continue

        packet = stream_manager.get_next_frame(camera_id, timeout=1.0)
        if packet is None:
            continue

        if _inference_engine is None:
            time.sleep(0.1)
            continue

        result = _inference_engine.predict(
            packet.frame, packet.frame_index, packet.camera_id
        )
        _result_processor.process(result, packet.frame)

        # Periyodik dataset aday kaydı
        now = time.time()
        if now - last_capture >= capture_interval:
            candidate_manager.process_frame(packet.frame, camera_id)
            last_capture = now


# ------------------------------------------------------------------ #
# Lifespan
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_engine, _result_processor, _inference_running, _inference_thread

    logger.info("=== Vitra Endüstriyel AI — Başlatılıyor ===")

    # MQTT ve Modbus
    mqtt_publisher.connect()
    modbus_writer.connect()

    # Model yükleme
    model_loaded = ensure_inference_model_loaded()
    if model_loaded:
        logger.info("Inference engine ready.")
    else:
        logger.warning("No production model loaded — inference disabled until model is promoted.")

    # Kamera
    camera_source = config.get("app", "camera", "source", default=0)
    camera_id: str = config.get("app", "pipeline", "default_camera_id", default="camera-0")
    stream_manager.add_camera(camera_id, camera_source)
    stream_manager.start()

    # Inference thread
    _inference_running = True
    _inference_thread = threading.Thread(
        target=_run_inference_loop, name="inference-loop", daemon=True
    )
    _inference_thread.start()

    logger.info("=== Servis hazır ===")
    yield

    # Shutdown
    _inference_running = False
    if _inference_thread:
        _inference_thread.join(timeout=5)
    stream_manager.stop()
    mqtt_publisher.disconnect()
    modbus_writer.disconnect()
    logger.info("=== Servis durduruldu ===")


# ------------------------------------------------------------------ #
# FastAPI app
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Vitra Endüstriyel AI",
    description="Sürekli Öğrenen Üretim Kalite Kontrol Sistemi — API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router'ları kaydet
app.include_router(camera.router, prefix="/api/camera", tags=["Camera"])
app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
app.include_router(labeling.router, prefix="/api/labeling", tags=["Labeling"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(model.router, prefix="/api/model", tags=["Model"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])

# Statik dashboard dosyaları
static_dir = Path("src/dashboard/static")
if static_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="dashboard")


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model_loader.is_loaded,
        "model_metadata": model_loader.model_metadata,
        "timestamp": time.time(),
    }


@app.get("/", tags=["Root"])
def root():
    return {"message": "Vitra Endüstriyel AI — Swagger: /docs"}
