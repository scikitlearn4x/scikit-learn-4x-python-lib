# ==================================================================
# Serialize PowerTransformer
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class PowerTransformerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_power_transformer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "_scaler", self.get_value_or_none(model, "_scaler"))
        self.add_field(fields, "lambdas_", self.get_value_or_none(model, "lambdas_"))
        self.add_field(fields, "method", self.get_value_or_none(model, "method"))
        self.add_field(fields, "standardize", self.get_value_or_none(model, "standardize"))

        self.add_n_features(fields, model)
        self.add_feature_names(fields, model)

        return fields
