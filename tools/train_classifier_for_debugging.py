import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp

RANDOM_STATE = 0

df = pd.read_csv('diabetes.csv')
X, y = df.drop('Outcome', axis=1), df['Outcome']
clf = GaussianNB()
clf.fit(X, y)
clf.predict_log_proba(X)

for field in dir(clf):
    value = getattr(clf, field)
    if '__' in field or callable(value):
        continue

    print(f'{field}: {value}')
