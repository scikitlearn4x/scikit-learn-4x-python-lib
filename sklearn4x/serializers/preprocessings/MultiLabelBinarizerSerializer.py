from sklearn4x.core.BaseSerializer import BaseSerializer


class MultiLabelBinarizerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_multilabel_binarizer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        if model.classes is not None:
            self.add_field(fields, 'classes', model.classes)

        self.add_field(fields, 'classes_', model.classes_.tolist())
        self.add_field(fields, '_cached_dict', model._cached_dict)
        self.add_field(fields, 'sparse_output', model.sparse_output)

        return fields