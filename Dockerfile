# Vitra Endüstriyel AI — Dockerfile
# Platform: linux/arm64 (Mac M4 uyumlu)
# Yalnızca inference + API servisi için — eğitim HOST'ta çalışır

FROM --platform=linux/arm64 python:3.11.9-slim

LABEL maintainer="Vitra AI Team"
LABEL description="Vitra Endüstriyel AI — Inference & API Service"

# Sistem bağımlılıkları (OpenCV headless için gerekli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Bağımlılıkları önce kopyala (Docker layer cache için)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Dizinleri oluştur (volume mount yoksa)
RUN mkdir -p data/raw_captures data/labeling_queue data/dataset \
             data/models/registry data/models/production logs

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# FastAPI servisi
EXPOSE 8000
ENV PYTHONPATH=/app
ENV CONFIG_DIR=/app/config

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
