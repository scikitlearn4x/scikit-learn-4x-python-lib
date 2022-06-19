import builtins

import sklearn.datasets
from sklearn.datasets import load_iris
from sklearn.preprocessing import *
from sklearn4x.sklearn4x import save_scikit_learn_model


def main():
    # targets = load_iris().data[0:15]
    targets = [[1.0, 0.5], [0.0, 0.4], [0.5, 0.2]]
    encoder = MinMaxScaler(clip=False)

    encoder.fit(targets)
    print(encoder.transform([[3.0, 1.5]]))
    # generate_code(encoder)

    print('')
    print('')
    print('====================================')
    print('')
    save_scikit_learn_model({'encoder': encoder}, 'test.skx')


def to_snake(cls):
    result = ''

    for ch in cls:
        if ch in 'QWERTYUIOPASDFGHJKLZXCVBNM':
            result += '_' + ch.lower()
        else:
            result += ch

    return result

def generate_code(encoder):
    cls = str(type(encoder))[:-2]
    cls = cls[cls.rindex('.')+1:]
    print("from sklearn4x.core.BaseSerializer import BaseSerializer")
    print('')
    print(f'class {cls}Serializer(BaseSerializer):')
    print('\tdef identifier(self):')
    print("\t\treturn 'pp" + to_snake(cls) + "'")
    print('')
    print('\tdef get_fields_to_be_serialized(self, model, version):')
    print('\t\tfields = []')
    print('')
    ignore = ['_repr_html_']
    for field in dir(encoder):
        if field in ignore or field.startswith('__'): continue
        attr = getattr(encoder, field)
        if str(attr) == "<class 'method'>" or '<bound method' in str(attr): continue
        print(f'\t\tself.add_field(fields, "{field}", model.{field})')

    print('')
    print('\t\treturn fields')


if __name__ == '__main__':
    main()
