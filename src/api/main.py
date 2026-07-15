"""
src/api/main.py
FastAPI uygulaması — inference servisi, dashboard API ve etiketleme endpoint'leri.
"""
import asyncio
import json
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
from src.registry.model_registry import ModelRegistry
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

_production_inference_engine: Optional[OnnxInference] = None
_result_processor: Optional[ResultProcessor] = None
_inference_thread: Optional[threading.Thread] = None
_inference_running = False
_camera_model_bindings: dict[str, str] = {}
_custom_model_loaders: dict[str, ModelLoader] = {}
_custom_inference_engines: dict[str, OnnxInference] = {}

CAMERA_MODEL_BINDINGS_PATH = Path(
    config.get("app", "data_dirs", "production_model", default="data/models/production")
) / "camera_model_bindings.json"
PRODUCTION_BINDING = "__production__"


# ------------------------------------------------------------------ #
# Alarm callback
# ------------------------------------------------------------------ #

def _alarm_callback(result) -> None:
    # Sadece sınıf adı "defect" ile başlayan tespitler alarm tetikler
    has_defect = any(d.class_name.startswith("defect") for d in result.detections)
    mqtt_publisher.publish_alarm(
        result.camera_id, result.detections, result.frame_index
    )
    modbus_writer.write_alarm(active=has_defect)


def _feedback_callback(frame: np.ndarray, camera_id: str) -> None:
    """Düşük confidence frame'lerini labeling queue'ya geri besler."""
    if not config.get("app", "pipeline", "low_confidence_feedback_enabled", default=False):
        return
    candidate_manager.process_frame(
        frame, camera_id, source="low_confidence_feedback", force_add=True
    )


def _ensure_result_processor() -> None:
    global _result_processor
    if _result_processor is None:
        _result_processor = ResultProcessor(
            alarm_callback=_alarm_callback,
            feedback_callback=_feedback_callback,
        )


def _load_camera_model_bindings() -> None:
    global _camera_model_bindings
    if not CAMERA_MODEL_BINDINGS_PATH.exists():
        _camera_model_bindings = {}
        return

    try:
        with open(CAMERA_MODEL_BINDINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_bindings = data.get("bindings", {}) if isinstance(data, dict) else {}
        _camera_model_bindings = {
            str(camera_id): str(binding)
            for camera_id, binding in raw_bindings.items()
            if binding
        }
    except Exception as exc:
        logger.warning("Failed to load camera model bindings: %s", exc)
        _camera_model_bindings = {}


def _persist_camera_model_bindings() -> None:
    CAMERA_MODEL_BINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "bindings": _camera_model_bindings,
    }
    with open(CAMERA_MODEL_BINDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_camera_model_binding(camera_id: str) -> str:
    return _camera_model_bindings.get(camera_id, PRODUCTION_BINDING)


def set_camera_model_binding(camera_id: str, binding: str) -> dict:
    normalized_binding = (binding or PRODUCTION_BINDING).strip() or PRODUCTION_BINDING
    if normalized_binding == PRODUCTION_BINDING:
        _camera_model_bindings.pop(camera_id, None)
    else:
        registry = ModelRegistry()
        metadata = registry.get_metadata(normalized_binding)
        model_path = registry.get_model_path(normalized_binding)
        if metadata is None or model_path is None:
            raise ValueError(f"Model version not found in registry: {normalized_binding}")
        _camera_model_bindings[camera_id] = normalized_binding
    _persist_camera_model_bindings()
    return get_camera_model_assignment(camera_id)


def get_camera_model_assignment(camera_id: str) -> dict:
    binding = get_camera_model_binding(camera_id)
    if binding == PRODUCTION_BINDING:
        return {
            "camera_id": camera_id,
            "binding": PRODUCTION_BINDING,
            "is_production": True,
            "model_metadata": model_loader.model_metadata,
        }

    metadata = ModelRegistry().get_metadata(binding) or {}
    return {
        "camera_id": camera_id,
        "binding": binding,
        "is_production": False,
        "model_metadata": metadata,
    }


def list_camera_model_assignments() -> dict[str, dict]:
    return {
        camera_id: get_camera_model_assignment(camera_id)
        for camera_id in stream_manager.list_camera_ids()
    }


def _ensure_custom_inference_engine(model_version: str) -> Optional[OnnxInference]:
    engine = _custom_inference_engines.get(model_version)
    if engine is not None:
        return engine

    registry = ModelRegistry()
    model_path = registry.get_model_path(model_version)
    if model_path is None:
        logger.warning("Registry model not found for camera binding: %s", model_version)
        return None

    loader = _custom_model_loaders.get(model_version)
    if loader is None:
        loader = ModelLoader()
        _custom_model_loaders[model_version] = loader

    if not loader.is_loaded and not loader.load_from_path(model_path):
        return None

    class_names: list[str] = config.get(
        "inference", "class_names", default=["defect_scratch", "defect_crack", "defect_dent", "ok"]
    )
    _ensure_result_processor()
    engine = OnnxInference(loader.session, class_names)
    _custom_inference_engines[model_version] = engine
    return engine


def get_inference_engine_for_camera(camera_id: str) -> Optional[OnnxInference]:
    binding = get_camera_model_binding(camera_id)
    if binding == PRODUCTION_BINDING:
        if not ensure_inference_model_loaded():
            return None
        return _production_inference_engine
    return _ensure_custom_inference_engine(binding)


def ensure_inference_model_loaded(force_reload: bool = False) -> bool:
    """Production modelini yükler ve inference engine'i senkronize eder."""
    global _production_inference_engine

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
    _production_inference_engine = OnnxInference(model_loader.session, class_names)
    _ensure_result_processor()
    return True


# ------------------------------------------------------------------ #
# Inference döngüsü
# ------------------------------------------------------------------ #

def _run_inference_loop() -> None:
    global _inference_running
    auto_capture_enabled: bool = config.get(
        "app", "pipeline", "auto_capture_to_queue_enabled", default=False
    )
    capture_interval: float = config.get("app", "pipeline", "capture_save_every_n_seconds", default=1500000.0)
    last_capture_by_camera: dict[str, float] = {}

    while _inference_running:
        camera_ids = stream_manager.list_camera_ids()
        if not camera_ids:
            time.sleep(0.2)
            continue

        processed_any = False
        for camera_id in camera_ids:
            packet = stream_manager.get_next_frame(camera_id, timeout=0.01)
            if packet is None:
                continue

            processed_any = True
            inference_engine = get_inference_engine_for_camera(packet.camera_id)
            if inference_engine is None or _result_processor is None:
                continue

            result = inference_engine.predict(
                packet.frame, packet.frame_index, packet.camera_id
            )
            annotated = _result_processor.process(result, packet.frame)
            stream_manager.set_latest_annotated_frame(packet.camera_id, annotated)

            if auto_capture_enabled:
                now = time.time()
                last_capture = last_capture_by_camera.get(packet.camera_id, 0.0)
                if now - last_capture >= capture_interval:
                    candidate_manager.process_frame(packet.frame, packet.camera_id)
                    last_capture_by_camera[packet.camera_id] = now

        if not processed_any:
            time.sleep(0.02)


# ------------------------------------------------------------------ #
# Lifespan
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _production_inference_engine, _result_processor, _inference_running, _inference_thread

    logger.info("=== Vitra Endüstriyel AI — Başlatılıyor ===")
    _load_camera_model_bindings()

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
        "camera_ids": stream_manager.list_camera_ids(),
        "camera_model_bindings": list_camera_model_assignments(),
        "timestamp": time.time(),
    }


@app.get("/", tags=["Root"])
def root():
    return {"message": "Vitra Endüstriyel AI — Swagger: /docs"}
