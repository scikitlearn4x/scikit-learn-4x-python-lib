import sklearn
from sklearn import datasets
from sklearn.naive_bayes import CategoricalNB
from sklearn4x.sklearn4x import save_scikit_learn_model
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# print('scikit-learn version: ' + sklearn.__version__)

support_probabilities = True

ds = datasets.load_iris()
X = ds.data
y = ds.target

train_data = X

classifier = CategoricalNB()
classifier.fit(train_data, y)

predictions = classifier.predict(X)

test_data = {
    "template_version": "classifiers_v1",
    "dataset_name": "iris",
    "configurations": [],
    "training_data": X,
    "predictions": predictions,

}

if support_probabilities:
    test_data["prediction_probabilities"] = classifier.predict_proba(X)
    test_data["prediction_log_probabilities"] = classifier.predict_log_proba(X)

save_scikit_learn_model({'classifier_to_test': classifier}, "/Users/yektaie/Documents/Generated Unit Tests/binaries/1.0.1/3.8/categorical_naive_bayes_simplest_base_case_without_customization_on_iris.skx", test_data)

