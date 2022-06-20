# ==================================================================
# Serialize MultiLabelBinarizer
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class MultiLabelBinarizerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_multilabel_binarizer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "classes_", self.get_value_or_none(model, "classes_"))
        self.add_field(fields, "_cached_dict", self.get_value_or_none(model, "_cached_dict"))

        return fields
