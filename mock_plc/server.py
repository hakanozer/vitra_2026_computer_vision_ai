from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
import logging

logging.basicConfig(level=logging.INFO)

# Proje sadece coil (alarm_coil_address: 0) kullanıyor (bkz. src/iot/modbus_client.py),
# bu yüzden yalnızca 'co' bloğu yeterli.
store = ModbusSlaveContext(co=ModbusSequentialDataBlock(0, [0] * 100))
context = ModbusServerContext(slaves=store, single=True)

print("Mock PLC Modbus TCP server starting on 0.0.0.0:502")
StartTcpServer(context=context, address=("0.0.0.0", 502))