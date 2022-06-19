from sklearn4x.core.BaseSerializer import BaseSerializer


class OrdinalEncoderSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_ordinal_encoder'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "_missing_indices", model._missing_indices)
        self.add_field(fields, "categories", model.categories)
        self.add_field(fields, "categories_", model.categories_)
        self.add_field(fields, "dtype", str(model.dtype))
        self.add_field(fields, "handle_unknown", model.handle_unknown)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "unknown_value", model.unknown_value)

        return fields
