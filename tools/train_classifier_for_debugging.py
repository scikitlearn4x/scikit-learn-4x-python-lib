import sklearn
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn4x.sklearn4x import save_scikit_learn_model

import scipy.sparse as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# print('scikit-learn version: ' + sklearn.__version__)

X = [[1.0, 0.5], [0.0, 0.4], [0.5, 0.2]]

preprocessing = MinMaxScaler()
preprocessing.fit(X)

transformed = preprocessing.transform([[3.0, 1.5]])
if isinstance(transformed, sp.csr_matrix):
    transformed = transformed.toarray()

test_data = {
    "template_version": "preprocessings_v1",
    "configurations": {'config_name': 'without range clipping', 'additional_import': '', 'class_argument': 'clip=False',
                       'custom_assertions': [], 'custom_transform_input': '[[3.0, 1.5]]',
                       'input': 'X = [[1.0, 0.5], [0.0, 0.4], [0.5, 0.2]]'},
    "raw": X,
    "transformed": transformed,
}

save_scikit_learn_model({'preprocessing_to_test': preprocessing},
                        "/Users/yektaie/Documents/Unit Test Generation/binaries/1.0.2/3.10/min_max_scaler_without_range_clipping.skx",
                        test_data)
