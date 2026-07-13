"""
src/iot/modbus_client.py
pymodbus tabanlı Modbus TCP istemcisi — PLC alarm coil'lerini yazar.
"""
import threading
import time
from typing import Optional

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from pymodbus.client import ModbusTcpClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    logger.warning("pymodbus not installed; Modbus disabled.")


class ModbusAlarmWriter:
    """
    PLC alarm coil'lerine Modbus TCP üzerinden yazar.
    Alarm coil: defect tespit edildiğinde 1, temiz frame'de 0 yazılır.
    """

    def __init__(self):
        self._enabled: bool = config.get("app", "modbus", "enabled", default=True)
        self._host: str = config.get("app", "modbus", "host", default="localhost")
        self._port: int = config.get("app", "modbus", "port", default=502)
        self._alarm_coil: int = config.get("app", "modbus", "alarm_coil_address", default=0)
        self._client: Optional[object] = None
        self._lock = threading.Lock()
        self._disabled_logged = False

    def connect(self) -> bool:
        if not self._enabled:
            if not self._disabled_logged:
                logger.info("Modbus disabled by config (app.modbus.enabled=false)")
                self._disabled_logged = True
            return False
        if not MODBUS_AVAILABLE:
            return False
        try:
            self._client = ModbusTcpClient(self._host, port=self._port, timeout=3)
            result = self._client.connect()
            if result:
                logger.info("Modbus connected to %s:%d", self._host, self._port)
            else:
                logger.warning("Modbus connection failed: %s:%d (PLC may not be running)", self._host, self._port)
            return result
        except Exception as exc:
            logger.warning("Modbus connect error: %s (PLC may not be running)", exc)
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.close()

    def write_alarm(self, active: bool) -> bool:
        """Alarm coil'ini yazar. active=True → defect var."""
        if not self._enabled:
            return False
        if not MODBUS_AVAILABLE:
            return False
        with self._lock:
            try:
                if self._client is None or not self._client.connected:
                    self.connect()
                if self._client and self._client.connected:
                    self._client.write_coil(self._alarm_coil, active)
                    return True
                else:
                    return False
            except Exception as exc:
                logger.warning("Modbus write_coil failed: %s (PLC may not be running)", exc)
                self._client = None
                return False
