"""
src/registry/model_registry.py
Eğitilen modelleri versiyonlu olarak saklar.
Her model: best.onnx + meta JSON (metrikler, dataset, tarih).
"""
import json
import shutil
import time
from pathlib import Path
from typing import Optional

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY_DIR = Path(
    config.get("app", "data_dirs", "model_registry", default="data/models/registry")
)


class ModelRegistry:
    """
    Modelleri data/models/registry/{version}/ altında saklar.

    Dizin yapısı:
        data/models/registry/
            v20240101_001/
                model.onnx
                model.pt
                metadata.json   ← metrikler, dataset snapshot, tarih
            v20240102_001/
                ...
    """

    def __init__(self):
        MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        pt_path: Path,
        onnx_path: Path,
        metrics: dict,
        dataset_version: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Yeni bir modeli registry'ye kaydeder.

        Parameters
        ----------
        pt_path         : ultralytics eğitiminden çıkan best.pt yolu
        onnx_path       : dışa aktarılmış best.onnx yolu
        metrics         : {"mAP50": 0.87, "precision": 0.91, "recall": 0.84, ...}
        dataset_version : Hangi dataset snapshot'ıyla eğitildi?
        version         : Zorunlu değil; verilmezse tarih bazlı üretilir

        Returns
        -------
        str : Atanan versiyon adı (örn. "v20240115_001")
        """
        if version is None:
            version = self._next_version()

        version_dir = MODEL_REGISTRY_DIR / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Dosyaları kopyala
        shutil.copy2(onnx_path, version_dir / "model.onnx")
        if pt_path.exists():
            shutil.copy2(pt_path, version_dir / "model.pt")

        # Meta kaydet
        metadata = {
            "version": version,
            "registered_at": time.time(),
            "registered_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_version": dataset_version,
            "metrics": metrics,
            "pt_source": str(pt_path),
            "onnx_source": str(onnx_path),
        }
        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "Model registered: version=%s mAP50=%.4f dataset=%s",
            version,
            metrics.get("mAP50", 0),
            dataset_version,
        )
        return version

    def get_metadata(self, version: str) -> Optional[dict]:
        """Belirtilen versiyonun meta verisini döner."""
        meta_path = MODEL_REGISTRY_DIR / version / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_model_path(self, version: str) -> Optional[Path]:
        """ONNX model dosyasının yolunu döner."""
        path = MODEL_REGISTRY_DIR / version / "model.onnx"
        return path if path.exists() else None

    def list_versions(self) -> list[dict]:
        """Tüm versiyonları kayıt tarihine göre sıralar."""
        versions = []
        for version_dir in sorted(MODEL_REGISTRY_DIR.iterdir()):
            if not version_dir.is_dir():
                continue
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    versions.append(json.load(f))
        return sorted(versions, key=lambda x: x.get("registered_at", 0), reverse=True)

    def _next_version(self) -> str:
        """Tarih ve sıra numarasından versiyon adı üretir: v20240115_001"""
        date_str = time.strftime("%Y%m%d")
        seq = 1
        while (MODEL_REGISTRY_DIR / f"v{date_str}_{seq:03d}").exists():
            seq += 1
        return f"v{date_str}_{seq:03d}"
