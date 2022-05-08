from sklearn4x.core.BinaryBuffer import BinaryBuffer


def main():
    buffer = BinaryBuffer()

    dictionary = {}

    buffer.append_dictionary(dictionary)
    data = buffer.to_buffer()
    data = list(data)

    print(data)


if __name__ == '__main__':
    main()
