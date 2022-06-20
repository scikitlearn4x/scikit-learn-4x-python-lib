import sklearn
from sklearn import datasets
from sklearn.preprocessing import RobustScaler
from sklearn4x.sklearn4x import save_scikit_learn_model
import sklearn.datasets as ds
import scipy.sparse as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# print('scikit-learn version: ' + sklearn.__version__)

X = ds.load_iris().data[0:15]

preprocessing = RobustScaler()
preprocessing.fit(X)

transformed = preprocessing.transform(X)
if isinstance(transformed, sp.csr_matrix):
    transformed = transformed.toarray()

test_data = {
    "template_version": "preprocessings_v1",
    "configurations": {'config_name': 'simple case without customization', 'additional_import': 'import sklearn.datasets as ds', 'class_argument': '', 'custom_assertions': [], 'custom_transform_input': 'X', 'input': 'X = ds.load_iris().data[0:15]'},
    "raw": X,
    "transformed": transformed,
}

save_scikit_learn_model({'preprocessing_to_test': preprocessing}, "/Users/yektaie/Documents/Generated Unit Tests/binaries/1.0.1/3.8/robust_scaler_simple_case_without_customization.skx", test_data)
