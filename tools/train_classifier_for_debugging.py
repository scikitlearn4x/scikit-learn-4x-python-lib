import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn4x.sklearn4x import save_scikit_learn_model
import sklearn.datasets as ds
import scipy.sparse as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# print('scikit-learn version: ' + sklearn.__version__)

X = ds.load_iris().data

preprocessing = OneHotEncoder(drop="first")
preprocessing.fit(X)

transformed = preprocessing.transform(X)
if isinstance(transformed, sp.csr_matrix):
    transformed = transformed.toarray()

test_data = {
    "template_version": "preprocessings_v1",
    "configurations": {'config_name': 'with drop first', 'additional_import': 'import sklearn.datasets as ds', 'class_argument': 'drop="first"', 'custom_assertions': [], 'custom_transform_input': 'X', 'input': 'X = ds.load_iris().data'},
    "raw": X,
    "transformed": transformed,
}

save_scikit_learn_model({'preprocessing_to_test': preprocessing}, "/Users/yektaie/Documents/Generated Unit Tests/binaries/1.0.2/3.9/one_hot_encoder_with_drop_first.skx", test_data)
