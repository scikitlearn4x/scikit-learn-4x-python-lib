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
    def test_append_float_nan(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(np.nan)

        expected = [0]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_nan(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(np.nan)

        expected = [0]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_float_pi_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(3.1415)

        expected = [1, 86, 14, 73, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_float_pi_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(-3.1415)

        expected = [1, 86, 14, 73, -64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_float_e_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(2.71828)

        expected = [1, 77, -8, 45, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_float_e_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_float(-2.71828)

        expected = [1, 77, -8, 45, -64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_pi_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(3.1415)

        expected = [1, 111, 18, -125, -64, -54, 33, 9, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_pi_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(-3.1415)

        expected = [1, 111, 18, -125, -64, -54, 33, 9, -64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_e_positive(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(2.71828)

        expected = [1, -112, -9, -86, -107, 9, -65, 5, 64]
        data = list(buffer.to_buffer())
        test_array_similarity(self, data, expected)

    def test_append_double_e_negative(self) -> None:
        buffer = BinaryBuffer()
        buffer.append_double(-2.71828)

        expected = [1, -112, -9, -86, -107, 9, -65, 5, -64]
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

    def test_append_none_string(self):
        buffer = BinaryBuffer()
        buffer.append_string(None)

        data = buffer.to_buffer()
        expected = [0]

        test_array_similarity(self, data, expected)

    def test_append_ascii_string(self):
        buffer = BinaryBuffer()
        buffer.append_string('test')

        data = buffer.to_buffer()
        expected = [1, 4, 0, 0, 0, 116, 101, 115, 116]

        test_array_similarity(self, data, expected)

    def test_append_utf8_string(self):
        buffer = BinaryBuffer()
        buffer.append_string('نمونه')

        data = buffer.to_buffer()
        expected = [1, 10, 0, 0, 0, 217, 134, 217, 133, 217, 136, 217, 134, 217, 135]

        test_array_similarity(self, data, expected)

    def test_append_simple_null_numpy_array(self):
        buffer = BinaryBuffer()
        buffer.append_numpy_array(None)

        data = buffer.to_buffer()
        expected = [0]

        test_array_similarity(self, data, expected)

    def test_append_simple_uint8_numpy_array(self):
        array = np.array([6, 7], dtype=np.uint8)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 1, 0, 0, 0, 17, 2, 0, 0, 0, 6, 7]
        test_array_similarity(self, data, expected)

    def test_append_simple_uint16_numpy_array(self):
        array = np.array([6, 7], dtype=np.uint16)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 1, 0, 0, 0, 18, 2, 0, 0, 0, 6, 0, 7, 0]
        test_array_similarity(self, data, expected)

    def test_append_simple_uint32_numpy_array(self):
        array = np.array([6, 7], dtype=np.uint32)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 1, 0, 0, 0, 20, 2, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0]
        test_array_similarity(self, data, expected)

    def test_append_simple_uint64_numpy_array(self):
        array = np.array([6, 7], dtype=np.uint64)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 1, 0, 0, 0, 24, 2, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0]
        test_array_similarity(self, data, expected)

    def test_append_simple_float_numpy_array(self):
        array = np.array([6, 7, np.nan], dtype=np.float32)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 1, 0, 0, 0, 32, 3, 0, 0, 0, 1, 0, 0, 192, 64, 1, 0, 0, 224, 64, 0]
        test_array_similarity(self, data, expected)

    def test_append_simple_double_numpy_array(self):
        array = np.array([6, 7, np.nan], dtype=np.float64)
        buffer = BinaryBuffer()
        buffer.append_numpy_array(array)

        data = buffer.to_buffer()
        expected = [1, 1, 0, 0, 0, 33, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 24, 64, 1, 0, 0, 0, 0, 0, 0, 28, 64, 0]
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

    def test_append_null_dictionary(self):
        buffer = BinaryBuffer()
        buffer.append_dictionary(None)

        data = buffer.to_buffer()
        expected = [0]

        test_array_similarity(self, data, expected)

    def test_append_dictionary(self):
        buffer = BinaryBuffer()

        dictionary = {
            'key_int': 15,
            'key_floating_point': 3.14,
            'key_string': 'This is a string',
            'key_list': [1, 3.14, 'another_string', {'sample': 'one', 'another': -6.84}, None],
            'key_dictionary': {
                'option_1': 'Java',
                'option_2': 'C#',
            },
            'null_key': None
        }

        buffer.append_dictionary(dictionary)
        data = buffer.to_buffer()
        expected = [1, 6, 0, 0, 0, 1, 7, 0, 0, 0, 107, 101, 121, 95, 105, 110, 116, 8, 15, 0, 0, 0, 0, 0, 0, 0, 1, 18, 0, 0, 0, 107, 101, 121, 95, 102, 108, 111, 97, 116, 105, 110, 103, 95, 112, 111, 105, 110, 116, 33, 1, 31,  133,  235, 81,  184, 30, 9, 64, 1, 10, 0, 0, 0, 107, 101, 121, 95, 115, 116, 114, 105, 110, 103, 48, 1, 16, 0, 0, 0, 84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 115, 116, 114, 105, 110, 103, 1, 8, 0, 0, 0, 107, 101, 121, 95, 108, 105, 115, 116, 64, 1, 5, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 33, 1, 31,  133,  235, 81,  184, 30, 9, 64, 48, 1, 14, 0, 0, 0, 97, 110, 111, 116, 104, 101, 114, 95, 115, 116, 114, 105, 110, 103, 65, 1, 2, 0, 0, 0, 1, 6, 0, 0, 0, 115, 97, 109, 112, 108, 101, 48, 1, 3, 0, 0, 0, 111, 110, 101, 1, 7, 0, 0, 0, 97, 110, 111, 116, 104, 101, 114, 33, 1, 92,  143,  194,  245, 40, 92, 27,  192, 16, 1, 14, 0, 0, 0, 107, 101, 121, 95, 100, 105, 99, 116, 105, 111, 110, 97, 114, 121, 65, 1, 2, 0, 0, 0, 1, 8, 0, 0, 0, 111, 112, 116, 105, 111, 110, 95, 49, 48, 1, 4, 0, 0, 0, 74, 97, 118, 97, 1, 8, 0, 0, 0, 111, 112, 116, 105, 111, 110, 95, 50, 48, 1, 2, 0, 0, 0, 67, 35, 1, 8, 0, 0, 0, 110, 117, 108, 108, 95, 107, 101, 121, 16]

        test_array_similarity(self, data, expected)
