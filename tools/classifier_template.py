from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn4x.sklearn4x import save_scikit_learn_model
import pandas as pd

support_probabilities = True

ds = datasets.load_breast_cancer()
X = ds.data
y = ds.target

train_data = X

classifier = GaussianNB()
classifier.fit(train_data, y)

predictions = classifier.predict(X)

test_data = {
    "dataset_name": "breast_cancer",
    "configurations": [],
    "training_data": X,
    "predictions": predictions,

}

if support_probabilities:
    test_data["prediction_probabilities"] = classifier.predict_proba(X)
    test_data["prediction_log_probabilities"] = classifier.predict_log_proba(X)

save_scikit_learn_model(classifier, "/Users/yektaie/Desktop/unit_tests/binaries/1.0/3.9/gaussian_naive_bayes_simplest_base_case_without_customization_on_breast_cancer.skx", test_data)

