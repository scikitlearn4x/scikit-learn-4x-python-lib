# ==================================================================
# Serialize MaxAbsScaler
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer


class MaxAbsScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_max_abs_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "scale_", self.get_value_or_none(model, "scale_"), version=version, min_version='0.17')
        self.add_field(fields, "max_abs_", self.get_value_or_none(model, "max_abs_"))
        self.add_field(fields, "n_samples_seen_", self.get_value_or_none(model, "n_samples_seen_"))

        self.add_n_features(fields, model)
        self.add_feature_names(fields, model)

        return fields
