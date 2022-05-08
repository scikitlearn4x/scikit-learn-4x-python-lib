import struct

LONG_SIZE = 4
INTEGER_SIZE = 4
SHORT_SIZE = 2
BYTE_SIZE = 1


class BinaryBuffer:
    def __init__(self):
        self.data = []

    def append_float(self, value: float):
        self.data.append(struct.pack('f', value))

    def append_double(self, value: float):
        self.data.append(struct.pack('d', value))

    def to_buffer(self):
        return b''.join(self.data)
