from abc import abstractmethod

from sklearn4x.core.BinaryBuffer import BinaryBuffer


class BinaryPackage:
    def __init__(self, buffer: BinaryBuffer):
        self.buffer = buffer

    @abstractmethod
    def create_file_header(self):
        pass

    @abstractmethod
    def append_serialized_model(self):
        pass

    @abstractmethod
    def add_additional_data(self, dictionary):
        pass

    @classmethod
    def default(cls, buffer: BinaryBuffer):
        return BinaryPackage(buffer)
