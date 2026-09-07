"""Injectable I2C bus abstraction shared by hardware drivers."""

from typing import Protocol, Self


class I2CBus(Protocol):
    """Subset of I2C operations required by Roadbot drivers."""

    def read_byte_data(self, address: int, register: int) -> int: ...

    def write_byte_data(self, address: int, register: int, value: int) -> None: ...

    def write_then_read(self, address: int, write: bytes, read_length: int) -> bytes: ...

    def close(self) -> None: ...


class SMBus2Bus:
    """Linux I2C adapter backed by smbus2."""

    def __init__(self, bus_number: int) -> None:
        from smbus2 import SMBus

        self._bus = SMBus(bus_number)

    def read_byte_data(self, address: int, register: int) -> int:
        return int(self._bus.read_byte_data(address, register))

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self._bus.write_byte_data(address, register, value)

    def write_then_read(self, address: int, write: bytes, read_length: int) -> bytes:
        from smbus2 import i2c_msg

        write_message = i2c_msg.write(address, write)
        read_message = i2c_msg.read(address, read_length)
        self._bus.i2c_rdwr(write_message, read_message)
        return bytes(read_message)

    def close(self) -> None:
        self._bus.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
