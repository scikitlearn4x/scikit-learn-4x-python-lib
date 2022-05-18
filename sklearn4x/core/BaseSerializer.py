from abc import *

from sklearn4x.core.BinaryBuffer import BinaryBuffer


class BaseSerializer:
    @abstractmethod
    def identifier(self):
        pass

    @abstractmethod
    def serialize_model(self, buffer: BinaryBuffer, model, version):
        pass

    def add_field(self, fields, name, value, version=None, min_version=None, max_version=None):
        if version is None and min_version is None and max_version is None:
            fields.append((name, value))

        if self.is_version_higher(version, min_version, True) and self.is_version_higher(max_version, version, False):
            fields.append((name, value))

    def is_version_higher(self, v1, v2, on_equal=True):
        if v1 is None:
            return True

        if v1 == v2:
            return on_equal

        c1 = v1.split('.')
        c2 = v2.split('.')

        max_length = max(len(c1), len(c2))
        self.__pad_with_zero(c1, max_length)
        self.__pad_with_zero(c2, max_length)

        for i in range(len(c1)):
            i1 = int(c1[i])
            i2 = int(c2[i])

            if i1 > i2:
                return True

        return False

    def __pad_with_zero(self, c, l):
        while len(c) < l:
            c.append('0')
