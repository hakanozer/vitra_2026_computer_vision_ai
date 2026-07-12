FROM --platform=linux/arm64 python:3.11.9-slim

WORKDIR /app

RUN pip install --no-cache-dir pymodbus==3.6.9

COPY server.py .

CMD ["python", "server.py"]