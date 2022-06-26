import sklearn
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn4x.sklearn4x import save_scikit_learn_model
import sklearn.datasets as ds
import scipy.sparse as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))

save_scikit_learn_model({'preprocessing_to_test': scaler}, '')
