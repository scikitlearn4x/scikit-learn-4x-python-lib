import sklearn
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB

from sklearn4x.core.BinaryBuffer import BinaryBuffer
from sklearn4x.core.BinaryPackage import BinaryPackage
from sklearn4x.serializers.naive_bayes.GaussianNaiveBayesSerializer import GaussianNaiveBayesSerializer

SERIALIZERS = [
    (GaussianNB, GaussianNaiveBayesSerializer())
]


def save_scikit_learn_model(models, path, additional_data=None):
    if not isinstance(models, list):
        models = [models]

    for i, model in enumerate(models):
        if not isinstance(model, BaseEstimator):
            raise Exception(f'The model provided at index {i} is not an scikit-learn BaseEstimator')

    content = []
    for model in models:
        for type, serializer in SERIALIZERS:
            if isinstance(model, type):
                content.append((model, serializer))
                break

    if len(content) != len(models):
        raise Exception(f'The models provided contains unsupported types.')

    # Prepare file header
    buffer = BinaryBuffer()
    package = BinaryPackage.default(buffer)
    package.create_file_header([serializer for model, serializer in content])

    # Append models
    for model, serializer in content:
        package.append_serialized_model(model, serializer)

    # Add additional data
    if additional_data is not None:
        package.add_additional_data(additional_data)

    # Save package
    package.save_to_file(path)
