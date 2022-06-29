# ==================================================================
# Serialize LabelBinarizer
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class LabelBinarizerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_label_binarizer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        classes_ = self.get_value_or_none(model, "classes_")
        if classes_ is not None:
            classes_ = classes_.tolist()
        self.add_field(fields, "classes_", classes_)

        self.add_field(fields, 'neg_label', model.neg_label)
        self.add_field(fields, 'pos_label', model.pos_label)
        self.add_field(fields, "y_type_", self.get_value_or_none(model, "y_type_"))

        return fields

    def append_field_to_buffer(self, buffer, name, value):
        buffer.append_string(name)
        if name == 'classes_':
            buffer.append_list(value)
        else:
            buffer.append_data(value)
