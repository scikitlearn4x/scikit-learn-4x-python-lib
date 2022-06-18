from sklearn4x.core.BaseSerializer import BaseSerializer


class QuantileTransformerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_quantile_transformer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "ignore_implicit_zeros", model.ignore_implicit_zeros)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "n_quantiles", model.n_quantiles)
        self.add_field(fields, "n_quantiles_", model.n_quantiles_)
        self.add_field(fields, "output_distribution", model.output_distribution)
        self.add_field(fields, "quantiles_", model.quantiles_)
        # self.add_field(fields, "random_state", model.random_state)
        self.add_field(fields, "references_", model.references_)
        self.add_field(fields, "subsample", model.subsample)

        return fields
