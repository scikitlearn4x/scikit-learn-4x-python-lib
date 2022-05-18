from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

from sklearn4x.sklearn4x import save_scikit_learn_model

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

classifier = GaussianNB()
classifier.fit(X, y)

predictions = classifier.predict(X)
prediction_probabilities = classifier.predict_proba(X)
prediction_log_probabilities = classifier.predict_log_proba(X)

test_data = {
    'dataset_name': 'diabetes',
    'configurations': [],
    'training_data': X,
    'predictions': predictions,
    'prediction_probabilities': prediction_probabilities,
    'prediction_log_probabilities': prediction_log_probabilities,
}

save_scikit_learn_model(classifier, '/User/yektaie/Desktop/gaussian_nb.skx', test_data)
