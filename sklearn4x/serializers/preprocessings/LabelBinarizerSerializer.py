from sklearn4x.core.BaseSerializer import BaseSerializer


class LabelBinarizerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_label_binarizer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, 'classes_', model.classes_.tolist())
        self.add_field(fields, 'neg_label', model.neg_label)
        self.add_field(fields, 'pos_label', model.pos_label)
        self.add_field(fields, 'sparse_input_', model.sparse_input_)
        self.add_field(fields, 'sparse_output', model.sparse_output)
        self.add_field(fields, 'y_type_', model.y_type_)

        return fields
