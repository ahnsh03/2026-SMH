# 재부팅 후: python3 ~/2026-SMH/check_i2c.py
import errno
from smbus2 import SMBus
b = SMBus(3)
for addr, name in [(0x40, "PCA9685"), (0x42, "INA219")]:
    try:
        b.read_byte_data(addr, 0x00)
        print(f"OK  0x{addr:02X} {name} 응답 정상")
    except OSError as e:
        print(f"FAIL 0x{addr:02X} {name}: errno={e.errno} ({errno.errorcode.get(e.errno)})")
b.close()
