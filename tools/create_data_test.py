import numpy as np


def get_java_type(array: np.ndarray):
    java_type = ''
    if array.dtype == np.float64:
        java_type = 'double'
    elif array.dtype == np.float32:
        java_type = 'float'
    elif array.dtype == np.int64:
        java_type = 'long'
    elif array.dtype == np.int32:
        java_type = 'int'
    elif array.dtype == np.int16:
        java_type = 'short'
    elif array.dtype == np.int8:
        java_type = 'byte'
    else:
        raise Exception()

    return java_type


def add_array_as_string(content, array):
    if len(array.shape) == 1:
        content.append('{')
        sep = ''
        for value in array:
            content.append(sep)
            content.append(str(value))
            sep = ', '
        content.append('}')
    else:
        content.append('{')
        sep = ''
        for value in array:
            content.append(sep)
            add_array_as_string(content, value)
            sep = ', '
        content.append('}')


def print_array(array, name):
    content = []
    content.append(get_java_type(array))
    content.append(''.join(['[]'] * len(array.shape)))
    content.append(' ')
    content.append(name)
    content.append('=')
    add_array_as_string(content, array)
    content.append(';')

    print(''.join(content))


if __name__ == '__main__':
    arr1 = np.array(np.random.random((5, 2)), dtype=np.float64)
    arr2 = np.array([1, 2], dtype=np.float64)

    print_array(arr1, 'arr1')
    print_array(arr2, 'arr2')
    print_array(arr2 - arr1, 'expected')
