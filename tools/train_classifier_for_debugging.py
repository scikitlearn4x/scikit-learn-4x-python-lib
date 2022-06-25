import sklearn
from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn4x.sklearn4x import save_scikit_learn_model

import scipy.sparse as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# print('scikit-learn version: ' + sklearn.__version__)

X = [{'sci-fi', 'thriller'}, {'comedy'}]

preprocessing = MultiLabelBinarizer()
preprocessing.fit(X)

transformed = preprocessing.transform(X)
if isinstance(transformed, sp.csr_matrix):
    transformed = transformed.toarray()

test_data = {
    "template_version": "preprocessings_v1",
    "configurations": {'config_name': 'with string labels', 'additional_import': '', 'class_argument': '', 'custom_assertions': [], 'custom_transform_input': 'X', 'input': "X = [{'sci-fi', 'thriller'}, {'comedy'}]"},
    "raw": X,
    "transformed": transformed,
}

save_scikit_learn_model({'preprocessing_to_test': preprocessing}, "/Users/yektaie/Documents/Unit Test Generation/binaries/0.21.1/3.6/multi_label_binarizer_with_string_labels.skx", test_data)
