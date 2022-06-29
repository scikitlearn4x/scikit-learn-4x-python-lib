from .naive_bayes import *
from sklearn.naive_bayes import *

from .preprocessings import *
from sklearn.preprocessing import *


LIST_OF_SERIALIZERS = {}


def add_sklearn_type(cls, serializer):
    try:
        classifier_type = eval(compile(cls, str(cls) + '_dynamic.py', 'eval'))
        LIST_OF_SERIALIZERS[classifier_type] = serializer
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
    add_sklearn_type('StandardScaler', StandardScalerSerializer())
    add_sklearn_type('MaxAbsScaler', MaxAbsScalerSerializer())
    add_sklearn_type('RobustScaler', RobustScalerSerializer())
    add_sklearn_type('Normalizer', NormalizerSerializer())
    add_sklearn_type('Binarizer', BinarizerSerializer())
    # add_sklearn_type('QuantileTransformer', QuantileTransformerSerializer())
    # add_sklearn_type('PowerTransformer', PowerTransformerSerializer())
    # add_sklearn_type('KBinsDiscretizer', KBinsDiscretizerSerializer())
    # add_sklearn_type('OneHotEncoder', OneHotEncoderSerializer())
    # add_sklearn_type('OrdinalEncoder', OrdinalEncoderSerializer())
    # add_sklearn_type('PolynomialFeatures', PolynomialFeaturesSerializer())
    # add_sklearn_type('SplineTransformer', SplineTransformerSerializer())


def load_list_of_serializers():
    load_naive_bayes_serializers()
    load_preprocessing_serializers()


if len(LIST_OF_SERIALIZERS) == 0:
    load_list_of_serializers()
