"""
src/registry/model_promoter.py
Yeni modeli production'a geçirme (promote) ve geri alma (rollback) mekanizması.
Production pointer: data/models/production/current_model.json
"""
import json
import time
from pathlib import Path
from typing import Optional

from src.registry.model_registry import MODEL_REGISTRY_DIR, ModelRegistry
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PRODUCTION_DIR = Path(
    config.get("app", "data_dirs", "production_model", default="data/models/production")
)
PRODUCTION_POINTER = PRODUCTION_DIR / "current_model.json"
PREVIOUS_POINTER = PRODUCTION_DIR / "previous_model.json"


class ModelPromoter:
    """
    Production model pointer yönetimi.

    Pointer dosyası (current_model.json) şu bilgileri içerir:
    {
      "version": "v20240115_001",
      "model_path": "data/models/registry/v20240115_001/model.onnx",
      "promoted_at": 1705312800,
      "metrics": { "mAP50": 0.87, ... }
    }

    Hot-swap: Pointer güncellendikten sonra API endpoint'i tetiklenirse
    çalışan inference servisi modeli yeniden başlatmaya gerek kalmadan yükler.
    """

    def __init__(self):
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        self._registry = ModelRegistry()

    def promote(self, version: str) -> bool:
        """
        Belirtilen versiyonu production'a geçirir.
        Mevcut production modeli 'previous_model.json' olarak yedeklenir.
        """
        metadata = self._registry.get_metadata(version)
        if metadata is None:
            logger.error("Version not found in registry: %s", version)
            return False

        model_path = self._registry.get_model_path(version)
        if model_path is None:
            logger.error("ONNX model file missing for version: %s", version)
            return False

        # Mevcut production'ı yedekle
        if PRODUCTION_POINTER.exists():
            import shutil
            shutil.copy2(PRODUCTION_POINTER, PREVIOUS_POINTER)

        # Yeni pointer yaz
        pointer = {
            "version": version,
            "model_path": str(model_path),
            "promoted_at": time.time(),
            "promoted_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": metadata.get("metrics", {}),
            "dataset_version": metadata.get("dataset_version", "unknown"),
        }
        with open(PRODUCTION_POINTER, "w", encoding="utf-8") as f:
            json.dump(pointer, f, indent=2, ensure_ascii=False)

        logger.info(
            "Model promoted to production: version=%s mAP50=%.4f",
            version,
            metadata.get("metrics", {}).get("mAP50", 0),
        )
        return True

    def rollback(self) -> bool:
        """
        Bir önceki production modeline geri döner.
        previous_model.json → current_model.json
        """
        if not PREVIOUS_POINTER.exists():
            logger.warning("No previous model to roll back to.")
            return False

        import shutil
        shutil.copy2(PREVIOUS_POINTER, PRODUCTION_POINTER)
        logger.info("Rolled back to previous production model.")
        return True

    def get_current_production(self) -> Optional[dict]:
        """Mevcut production pointer bilgilerini döner."""
        if not PRODUCTION_POINTER.exists():
            return None
        with open(PRODUCTION_POINTER, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_previous_production(self) -> Optional[dict]:
        """Önceki production pointer bilgilerini döner (rollback hedefi)."""
        if not PREVIOUS_POINTER.exists():
            return None
        with open(PREVIOUS_POINTER, "r", encoding="utf-8") as f:
            return json.load(f)
