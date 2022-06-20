# ==================================================================
# Serialize QuantileTransformer
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class QuantileTransformerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_quantile_transformer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "n_quantiles_", self.get_value_or_none(model, "n_quantiles_"))
        self.add_field(fields, "quantiles_", self.get_value_or_none(model, "quantiles_"))
        self.add_field(fields, "references_", self.get_value_or_none(model, "references_"))
        self.add_field(fields, "ignore_implicit_zeros", self.get_value_or_none(model, "ignore_implicit_zeros"))
        self.add_field(fields, "output_distribution", self.get_value_or_none(model, "output_distribution"))
        self.add_field(fields, "subsample", self.get_value_or_none(model, "subsample"))

        self.add_n_features(fields, model)
        self.add_feature_names(fields, model)

        return fields
