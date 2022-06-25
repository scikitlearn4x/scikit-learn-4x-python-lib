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

        classes_ = self.get_value_or_none(model, "classes_")
        if classes_ is not None:
            classes_ = classes_.tolist()
        self.add_field(fields, "classes_", classes_)

        return fields

    def append_field_to_buffer(self, buffer, name, value):
        buffer.append_string(name)
        if name == 'classes_':
            buffer.append_list(value)
        else:
            buffer.append_data(value)
