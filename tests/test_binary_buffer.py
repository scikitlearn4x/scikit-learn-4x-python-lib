from unittest import TestCase

from typing import List

from sklearn4x.core.BinaryBuffer import BinaryBuffer


def test_array_similarity(self, actual: List[int], expected: List[int]) -> None:
    """
    Test if two list are the same or not.

    :param self: The instance to the unittest class.
    :param actual: The data to be tested.
    :param expected: The expected correct value.

    :return: None
    """
    self.assertEqual(len(expected), len(actual))
    for i, value in enumerate(actual):
        e = expected[i]
        # The original class for binary buffer was in Java that doesn't support unsigned values
        # Because of that, a conversion of the negative numbers is needed.
        if e < 0:
            e = 256 + e
        self.assertEqual(e, value)


class TestBinaryBuffer(TestCase):
    def test_append_float_pi_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(3.1415)

        expected = [86, 14, 73, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_float_pi_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(-3.1415)

        expected = [86, 14, 73, -64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_float_e_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(2.71828)

        expected = [77, -8, 45, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_float_e_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(-2.71828)

        expected = [77, -8, 45, -64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_pi_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(3.1415)

        expected = [111, 18, -125, -64, -54, 33, 9, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_pi_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(-3.1415)

        expected = [111, 18, -125, -64, -54, 33, 9, -64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_e_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(2.71828)

        expected = [-112, -9, -86, -107, 9, -65, 5, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_e_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(-2.71828)

        expected = [-112, -9, -86, -107, 9, -65, 5, -64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)


