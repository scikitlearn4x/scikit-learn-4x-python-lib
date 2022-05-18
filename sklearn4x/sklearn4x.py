import sklearn
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB

from sklearn4x.core.BinaryBuffer import BinaryBuffer
from sklearn4x.core.BinaryPackage import BinaryPackage
from sklearn4x.serializers.naive_bayes.GaussianNaiveBayesSerializer import GaussianNaiveBayesSerializer

SERIALIZERS = [
    (GaussianNB, GaussianNaiveBayesSerializer)
]


def save_scikit_learn_model(model, path, additional_data=None):
    if not isinstance(model, BaseEstimator):
        raise Exception('The model provided is not an scikit-learn BaseEstimator')

    found = False
    for type, serializer in SERIALIZERS:
        if isinstance(model, type):
            found = True

            # Prepare file header
            buffer = BinaryBuffer()
            package = BinaryPackage.default(buffer)
            package.create_file_header([serializer])

            # Append models
            package.append_serialized_model(model, serializer)

            # Add additional data
            if additional_data is not None:
                package.add_additional_data(additional_data)

            # Save package
            package.save_to_file(path)

    if not found:
        raise Exception('The model provided is not supported')


