from abc import *


class BaseSerializer:
    @abstractmethod
    def serialize_model(self, model, version):
        pass
