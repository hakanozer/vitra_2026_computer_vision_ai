"""
src/inference/model_loader.py
ONNX model dosyasını yükleyen ve yönetemeden sorumlu modül.
Production model pointer dosyasını izler (hot-swap desteği).
"""
import json
import threading
from pathlib import Path
from typing import Optional

import onnxruntime as ort

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY_DIR = Path(
    config.get("app", "data_dirs", "model_registry", default="data/models/registry")
)
PRODUCTION_DIR = Path(
    config.get("app", "data_dirs", "production_model", default="data/models/production")
)
PRODUCTION_POINTER = PRODUCTION_DIR / "current_model.json"


def _get_execution_providers() -> list[str]:
    """
    Çalışma ortamına göre en iyi execution provider listesini döner.
    Mac M4: CoreMLExecutionProvider → CPUExecutionProvider
    Linux/NVIDIA: CUDAExecutionProvider → CPUExecutionProvider (kavramsal)
    Varsayılan güvenli: CPUExecutionProvider
    """
    available = ort.get_available_providers()
    preferred = []
    if "CoreMLExecutionProvider" in available:
        preferred.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in available:
        preferred.append("CPUExecutionProvider")
    if not preferred:
        preferred = ["CPUExecutionProvider"]
    logger.info("ONNX execution providers: %s", preferred)
    return preferred


class ModelLoader:
    """
    ONNX modelini yükler ve InferenceSession döner.
    Hot-swap için reload() metodu sağlar.
    """

    def __init__(self):
        self._session: Optional[ort.InferenceSession] = None
        self._model_path: Optional[Path] = None
        self._lock = threading.RLock()
        self._metadata: dict = {}

    def load_production_model(self) -> bool:
        """
        data/models/production/current_model.json pointer dosyasını okuyup
        işaret ettiği ONNX modelini yükler.
        """
        if not PRODUCTION_POINTER.exists():
            logger.warning(
                "No production model pointer found at %s. "
                "Run 'scripts/promote.py' to set a production model.",
                PRODUCTION_POINTER,
            )
            return False

        with open(PRODUCTION_POINTER, "r", encoding="utf-8") as f:
            pointer = json.load(f)

        model_path = Path(pointer.get("model_path", ""))
        if not model_path.exists():
            logger.error("Production model file not found: %s", model_path)
            return False

        return self._load(model_path, pointer)

    def load_from_path(self, model_path: Path) -> bool:
        """Belirtilen ONNX dosyasını doğrudan yükler (test/geliştirme için)."""
        return self._load(model_path, {})

    def _load(self, model_path: Path, metadata: dict) -> bool:
        """
        ONNX InferenceSession oluşturur.
        Lock altında yapılır — hot-swap sırasında race condition önlenir.
        """
        providers = _get_execution_providers()
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            new_session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_opts,
                providers=providers,
            )
            with self._lock:
                self._session = new_session
                self._model_path = model_path
                self._metadata = metadata
            logger.info("Model loaded: %s (providers=%s)", model_path.name, providers)
            return True
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_path, exc)
            return False

    def reload(self) -> bool:
        """
        Production pointer'ı yeniden okuyup modeli hot-swap eder.
        Inference sırasında çağrılabilir — lock sayesinde atomik.
        """
        logger.info("Hot-swap: reloading production model …")
        return self.load_production_model()

    @property
    def session(self) -> Optional[ort.InferenceSession]:
        with self._lock:
            return self._session

    @property
    def model_metadata(self) -> dict:
        with self._lock:
            return dict(self._metadata)

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._session is not None
