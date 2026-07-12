"""
src/utils/config_loader.py
YAML tabanlı konfigürasyon yükleme ve erişim yardımcısı.
"""
import os
from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "config"))


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_get(data: dict, *keys: str, default: Any = None) -> Any:
    """Nokta notasyonuyla iç içe dict'e erişim."""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
        if data is None:
            return default
    return data


class AppConfig:
    """Singleton benzeri config nesnesi; uygulama boyunca tek instance."""

    _instance: "AppConfig | None" = None
    _data: dict = {}

    def __new__(cls) -> "AppConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_all()
        return cls._instance

    def _load_all(self) -> None:
        config_files = {
            "app": "app_config.yaml",
            "inference": "inference_config.yaml",
            "training": "training_config.yaml",
            "quality": "quality_config.yaml",
        }
        for key, filename in config_files.items():
            path = _CONFIG_DIR / filename
            if path.exists():
                self._data[key] = _load_yaml(path)
            else:
                self._data[key] = {}

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Örnek: config.get("inference", "confidence_threshold", default=0.5)
        """
        return _deep_get(self._data, *keys, default=default)

    def reload(self) -> None:
        """Runtime'da config yeniden yükleme (hot-reload desteği)."""
        self._load_all()


# Modüller şunu import eder: from src.utils.config_loader import AppConfig
config = AppConfig()
