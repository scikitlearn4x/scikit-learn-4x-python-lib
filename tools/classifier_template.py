from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn4x.sklearn4x import save_scikit_learn_model
import pandas as pd

ds = datasets.load_iris()
X = ds.data
y = ds.target

train_data = pd.DataFrame(data=X, index=None, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], dtype=X.dtype, copy=False)

classifier = GaussianNB()
classifier.fit(train_data, y)

predictions = classifier.predict(X)
prediction_probabilities = classifier.predict_proba(X)
prediction_log_probabilities = classifier.predict_log_proba(X)

test_data = {
    "training_data": X,
    "predictions": predictions,
    "prediction_probabilities": prediction_probabilities,
    "prediction_log_probabilities": prediction_log_probabilities,
}

print(f'First data point prediction: {predictions[0]}')
print(f'First data point probabilities: {prediction_probabilities[0, 0]:.3f}, {prediction_probabilities[0, 1]:.3f}')
print(f'First data point log probabilities: {prediction_log_probabilities[0, 0]:.3f}, {prediction_log_probabilities[0, 1]:.3f}')

save_scikit_learn_model(classifier, '/Users/yektaie/Desktop/usage.skx', test_data)