"""
src/iot/mqtt_client.py
paho-mqtt tabanlı MQTT yayıncısı — tespit alarmlarını broker'a gönderir.
"""
import json
import threading
import time
from typing import Optional

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed; MQTT disabled.")


class MqttPublisher:
    """
    Tespit alarmlarını MQTT broker'a yayınlar.
    Topic: {base_topic}/camera/{camera_id}/alarm
    """

    def __init__(self):
        self._host: str = config.get("app", "mqtt", "host", default="localhost")
        self._port: int = config.get("app", "mqtt", "port", default=1883)
        self._base_topic: str = config.get("app", "mqtt", "base_topic", default="vitra/ai")
        self._client: Optional[object] = None
        self._connected = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        if not MQTT_AVAILABLE:
            return False
        try:
            self._client = mqtt.Client(client_id="vitra-ai-inference")
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.connect(self._host, self._port, keepalive=60)
            self._client.loop_start()
            # Bağlantı kurulana kadar max 3 saniye bekle
            deadline = time.time() + 3
            while not self._connected and time.time() < deadline:
                time.sleep(0.1)
            return self._connected
        except Exception as exc:
            logger.error("MQTT connect failed: %s", exc)
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()

    def publish_alarm(self, camera_id: str, detections: list, frame_index: int) -> None:
        """Tespit alarmını MQTT'ye yayınlar."""
        if not self._connected or not MQTT_AVAILABLE:
            return
        topic = f"{self._base_topic}/camera/{camera_id}/alarm"
        payload = {
            "camera_id": camera_id,
            "frame_index": frame_index,
            "timestamp": time.time(),
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox_xyxy,
                }
                for d in detections
            ],
        }
        try:
            self._client.publish(topic, json.dumps(payload), qos=1)
        except Exception as exc:
            logger.warning("MQTT publish failed: %s", exc)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to %s:%d", self._host, self._port)
        else:
            logger.warning("MQTT connection refused: rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        logger.warning("MQTT disconnected: rc=%d", rc)
