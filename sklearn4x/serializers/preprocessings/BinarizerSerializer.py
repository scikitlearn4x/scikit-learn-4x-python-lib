# ==================================================================
# Serialize Binarizer
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class BinarizerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_binarizer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "threshold", self.get_value_or_none(model, "threshold"))

        self.add_n_features(fields, model)
        self.add_feature_names(fields, model)

        return fields
