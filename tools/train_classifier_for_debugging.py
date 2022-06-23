import sklearn
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn4x.sklearn4x import save_scikit_learn_model

import scipy.sparse as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# print('scikit-learn version: ' + sklearn.__version__)

X = ['a', 'a', 'c', 'd', 'a', 'd', 'b', 'b', 'c']

preprocessing = LabelEncoder()
preprocessing.fit(X)

transformed = preprocessing.transform(X)
if isinstance(transformed, sp.csr_matrix):
    transformed = transformed.toarray()

test_data = {
    "template_version": "preprocessings_v1",
    "configurations": {'config_name': 'with string labels',
                       'additional_import': '', 'class_argument': '',
                       'custom_assertions': [],
                       'custom_transform_input': 'X', 'input': "X = ['a', 'a', 'c', 'd', 'a', 'd', 'b', 'b', 'c']"},
    "raw": X,
    "transformed": transformed,
}

save_scikit_learn_model({'preprocessing_to_test': preprocessing}, "/Users/yektaie/Documents/Generated Unit Tests/binaries/1.0.2/3.10/label_encoder_with_string_labels.skx", test_data)
