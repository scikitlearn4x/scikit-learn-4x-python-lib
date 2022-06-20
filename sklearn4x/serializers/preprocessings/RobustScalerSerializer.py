# ==================================================================
# Serialize RobustScaler
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class RobustScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_robust_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "center_", self.get_value_or_none(model, "center_"))
        self.add_field(fields, "with_scaling", self.get_value_or_none(model, "with_scaling"))
        self.add_field(fields, "with_centering", self.get_value_or_none(model, "with_centering"))
        self.add_field(fields, "unit_variance", self.get_value_or_none(model, "unit_variance"))
        self.add_field(fields, "quantile_range", self.get_value_or_none(model, "quantile_range"))
        self.add_field(fields, "scale_", self.get_value_or_none(model, "scale_"), version=version, min_version='0.17')

        self.add_n_features(fields, model)
        self.add_feature_names(fields, model)

        return fields


