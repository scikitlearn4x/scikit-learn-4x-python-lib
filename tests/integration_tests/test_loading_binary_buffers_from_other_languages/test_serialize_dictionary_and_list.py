from sklearn4x.core.BinaryBuffer import BinaryBuffer


def main():
    buffer = BinaryBuffer()

    dictionary = {
        'key_int': 15,
        'key_floating_point': 3.14,
        'key_string': 'This is a string',
        'key_list': [1, 3.14, 'another_string', {'sample': 'one', 'another': -6.84}],
        'key_dictionary': {
            'option_1': 'Java',
            'option_2': 'C#',
        }
    }

    buffer.append_dictionary(dictionary)
    data = buffer.to_buffer()
    data = list(data)

    print(data)


if __name__ == '__main__':
    main()
