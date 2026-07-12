"""
src/quality/image_quality.py
Görüntü kalite analizi — blur, pozlama ve kadraj kontrolleri.
Eşik altındaki frame'ler dataset adayı olmaz.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityReport:
    """Bir frame için hesaplanan kalite metrikleri."""

    blur_score: float        # Laplacian varyansı — yüksek = keskin
    brightness: float        # Ortalama piksel değeri (0–255)
    contrast: float          # Standart sapma
    is_acceptable: bool      # Tüm eşikler geçildi mi?
    rejection_reason: str    # Reddedildiyse neden


class ImageQualityAnalyzer:
    """
    Blur, pozlama ve kontrast kontrolü yapar.
    Her frame için QualityReport üretir.

    Neden önemli?
    -------------
    Kötü kaliteli frame'lerin dataset'e girmesi modeli yanıltır:
    - Bulanık frame: model ince hasar detaylarını öğrenemez
    - Aşırı parlak/karanlık: renk ve doku bilgisi kaybolur
    - Düşük kontrast: kenarlar belirsizleşir, bbox doğruluğu düşer
    """

    def __init__(self):
        self._blur_threshold: float = config.get(
            "quality", "blur_threshold", default=100.0
        )
        self._min_brightness: float = config.get(
            "quality", "min_brightness", default=30.0
        )
        self._max_brightness: float = config.get(
            "quality", "max_brightness", default=225.0
        )
        self._min_contrast: float = config.get(
            "quality", "min_contrast", default=20.0
        )

    def analyze(self, frame: np.ndarray) -> QualityReport:
        """
        BGR frame alır, QualityReport döner.

        Algoritma adımları:
        1. Gri tonlamaya çevir (renk bilgisi kalite hesabında gürültü yaratır)
        2. Laplacian filtresi ile kenar yoğunluğunu ölç → blur skoru
        3. Ortalama parlaklık ve standart sapma hesapla
        4. Her metriği eşikle karşılaştır
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur tespiti: Laplacian filtresi köşeleri ve kenarları vurgular.
        # Varyans düşükse → tüm değerler birbirine yakın → görüntü bulanık.
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Parlaklık ve kontrast: mean = ortalama piksel, std = dağılım genişliği
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Karar mantığı
        rejection_reason = ""
        is_acceptable = True

        if blur_score < self._blur_threshold:
            is_acceptable = False
            rejection_reason = f"blur_score={blur_score:.1f} < threshold={self._blur_threshold}"
        elif brightness < self._min_brightness:
            is_acceptable = False
            rejection_reason = f"brightness={brightness:.1f} < min={self._min_brightness}"
        elif brightness > self._max_brightness:
            is_acceptable = False
            rejection_reason = f"brightness={brightness:.1f} > max={self._max_brightness}"
        elif contrast < self._min_contrast:
            is_acceptable = False
            rejection_reason = f"contrast={contrast:.1f} < min={self._min_contrast}"

        if not is_acceptable:
            logger.debug("Frame rejected: %s", rejection_reason)

        return QualityReport(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            is_acceptable=is_acceptable,
            rejection_reason=rejection_reason,
        )
