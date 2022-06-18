from .naive_bayes import *
from sklearn.naive_bayes import *

from .preprocessings import *
from sklearn.preprocessing import *


LIST_OF_SERIALIZERS = []


def add_sklearn_type(cls, serializer):
    try:
        classifier_type = eval(compile(cls, f'{cls}_dynamic.py', 'eval'))
        LIST_OF_SERIALIZERS.append((classifier_type, serializer))
    except NameError as ex:
        # Ignore the type, it is not supported by this version of sklearn.
        pass


def load_naive_bayes_serializers():
    add_sklearn_type('GaussianNB', GaussianNaiveBayesSerializer())
    add_sklearn_type('BernoulliNB', BernoulliNaiveBayesSerializer())
    add_sklearn_type('MultinomialNB', MultinomialNaiveBayesSerializer())
    add_sklearn_type('ComplementNB', ComplementNaiveBayesSerializer())
    add_sklearn_type('CategoricalNB', CategoricalNaiveBayesSerializer())


def load_preprocessing_serializers():
    add_sklearn_type('LabelEncoder', LabelEncoderSerializer())
    add_sklearn_type('LabelBinarizer', LabelBinarizerSerializer())
    add_sklearn_type('MultiLabelBinarizer', MultiLabelBinarizerSerializer())
    add_sklearn_type('MinMaxScaler', MinMaxScalerSerializer())


def load_list_of_serializers():
    load_naive_bayes_serializers()
    load_preprocessing_serializers()


if len(LIST_OF_SERIALIZERS) == 0:
    load_list_of_serializers()
