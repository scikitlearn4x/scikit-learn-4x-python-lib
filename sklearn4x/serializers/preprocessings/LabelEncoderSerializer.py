# ==================================================================
# Serialize LabelEncoder
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class LabelEncoderSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_label_encoder'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "classes_", self.get_value_or_none(model, "classes_"))

        return fields
