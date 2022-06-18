from sklearn4x.core.BaseSerializer import BaseSerializer


class NormalizerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_normalizer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "norm", model.norm)

        return fields
