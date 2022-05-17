import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp
from create_data_test import *

RANDOM_STATE = 0

df = pd.read_csv('diabetes.csv')
X, y = df.drop('Outcome', axis=1), df['Outcome']
clf = GaussianNB()
clf.fit(X, y)

print_array(X.to_numpy(), 'x')
print_array(clf.class_count_, 'class_count_')
print_array(clf.class_prior_, 'class_prior_')
print_array(clf.classes_, 'classes_')
print_array(clf.sigma_, 'sigma_')
print_array(clf.theta_, 'theta_')
clf.predict_log_proba(X)
#
# for field in dir(clf):
#     value = getattr(clf, field)
#     if '__' in field or callable(value):
#         continue
#
#     print(f'{field}: {value}')
