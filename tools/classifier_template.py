from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn4x.sklearn4x import save_scikit_learn_model

support_probabilities = True

ds = datasets.load_iris()
X = ds.data
y = ds.target

classifier = GaussianNB()
classifier.fit(X, y)

predictions = classifier.predict(X)

test_data = {
    "dataset_name": "diabetes",
    "configurations": [],
    "training_data": X,
    "predictions": predictions,
}

if support_probabilities:
    test_data["prediction_probabilities"] = classifier.predict_proba(X)
    test_data["prediction_log_probabilities"] = classifier.predict_log_proba(X)


