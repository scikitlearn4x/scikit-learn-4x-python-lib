import sklearn
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn4x.sklearn4x import save_scikit_learn_model

import scipy.sparse as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# print('scikit-learn version: ' + sklearn.__version__)

X = [1, 2, 2, 6]

preprocessing = LabelEncoder()
preprocessing.fit(X)

transformed = preprocessing.transform(X)
if isinstance(transformed, sp.csr_matrix):
    transformed = transformed.toarray()

test_data = {
    "template_version": "preprocessings_v1",
    "configurations": {'config_name': 'with int labels', 'additional_import': '', 'class_argument': '',
                       'custom_assertions': [], 'custom_transform_input': 'X', 'input': 'X = [1, 2, 2, 6]'},
    "raw": X,
    "transformed": transformed,
}

save_scikit_learn_model({'preprocessing_to_test': preprocessing},
                        "/Users/yektaie/Documents/Unit Test Generation/binaries/0.20.0/3.5/label_encoder_with_int_labels.skx",
                        test_data)
