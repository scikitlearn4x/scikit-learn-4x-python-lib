from sklearn4x.core.BaseSerializer import BaseSerializer


class LabelEncoderSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_label_encoder'

    def get_fields_to_be_serialized(self, model, version):
        fields = []
        self.add_field(fields, 'classes_', model.classes_.tolist())

        return fields
