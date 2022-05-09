from unittest import TestCase

from typing import List

import numpy as np

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

    def test_append_byte_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_byte(23)

        expected = [23]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_byte_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_byte(-8)

        expected = [248]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_short_values(self):
        examples = [
            (10, [10, 0]),
            (567, [55, 2]),
            (16000, [-128, 62]),
            (-16000, [-128, -63]),
        ]

        for number, expected_output in examples:
            buffer = BinaryBuffer()
            buffer.append_short(number)

            data = list(buffer.to_buffer())
            test_array_similarity(self, data, expected_output)

    def test_append_int_values(self):
        examples = [
            (10, [10, 0, 0, 0]),
            (567, [55, 2, 0, 0]),
            (16000, [-128, 62, 0, 0]),
            (59, [59, 0, 0, 0]),
            (-59, [-59, -1, -1, -1]),
            (300, [44, 1, 0, 0]),
            (-300, [-44, -2, -1, -1]),
            (2000000000, [0, -108, 53, 119]),
            (-2000000000, [0, 108, -54, -120]),
        ]

        for number, expected_output in examples:
            buffer = BinaryBuffer()
            buffer.append_int(number)

            data = list(buffer.to_buffer())
            test_array_similarity(self, data, expected_output)

    def test_append_long_values(self):
        examples = [
            (10, [10, 0, 0, 0, 0, 0, 0, 0]),
            (567, [55, 2, 0, 0, 0, 0, 0, 0]),
            (16000, [-128, 62, 0, 0, 0, 0, 0, 0]),
            (59, [59, 0, 0, 0, 0, 0, 0, 0]),
            (-59, [-59, -1, -1, -1, -1, -1, -1, -1]),
            (300, [44, 1, 0, 0, 0, 0, 0, 0]),
            (-300, [-44, -2, -1, -1, -1, -1, -1, -1]),
            (2000000000, [0, -108, 53, 119, 0, 0, 0, 0]),
            (-2000000000, [0, 108, -54, -120, -1, -1, -1, -1]),
            (200000000000000000, [0, 0, 20, -69, -16, -118, -58, 2]),
            (-200000000000000000, [0, 0, -20, 68, 15, 117, 57, -3]),
        ]

        for number, expected_output in examples:
            buffer = BinaryBuffer()
            buffer.append_long(number)

            data = list(buffer.to_buffer())
            test_array_similarity(self, data, expected_output)

    def test_append_simple_null_numpy_array(self):
        buffer = BinaryBuffer()
        buffer.append_numpy_array(None)

        data = buffer.to_buffer()
        expected = [0]

        test_array_similarity(self, data, expected)

    def test_append_simple_numpy_array(self):
        array = np.array([6, 7], dtype=np.uint8)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 1, 0, 0, 0, 17, 2, 0, 0, 0, 6, 7]
        test_array_similarity(self, data, expected)

    def test_append_vertical_numpy_array(self):
        array = np.array([[6], [7]], dtype=np.uint8)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 2, 0, 0, 0, 17, 2, 0, 0, 0, 1, 0, 0, 0, 6, 7]
        test_array_similarity(self, data, expected)

    def test_append_vertical_numpy_array_of_type_int(self):
        array = np.array([[6], [7]], dtype=np.int32)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 2, 0, 0, 0, 4, 2, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0]
        test_array_similarity(self, data, expected)

    def test_append_3dim_tensor_numpy_array_of_type_int(self):
        array = np.array([[[6], [7]], [[8], [9]]], dtype=np.int32)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 3, 0, 0, 0, 4, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0]
        test_array_similarity(self, data, expected)