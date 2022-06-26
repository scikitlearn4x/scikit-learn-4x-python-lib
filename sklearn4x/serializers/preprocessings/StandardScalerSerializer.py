# ==================================================================
# Serialize StandardScaler
#
# Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
# ==================================================================
from sklearn4x.core.BaseSerializer import BaseSerializer
import numpy as np

class StandardScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_standard_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "scale_", self.get_value_or_none(model, "scale_"), version=version, min_version='0.17')
        self.add_field(fields, "mean_", self.get_value_or_none(model, "mean_"))
        self.add_field(fields, "var_", self.get_value_or_none(model, "var_"))
        self.add_field(fields, "with_mean", self.get_value_or_none(model, "with_mean"))
        self.add_field(fields, "with_std", self.get_value_or_none(model, "with_std"))

        n_samples_seen_ = self.get_value_or_none(model, 'n_samples_seen_')
        if isinstance(n_samples_seen_, np.int64):
            feature_count = None
            if model.scale_ is not None:
                feature_count = model.scale_.shape[0]
            elif model.mean_ is not None:
                feature_count = model.mean_.shape[0]

            n_samples_seen_ = [n_samples_seen_] * feature_count
            n_samples_seen_ = np.array(n_samples_seen_)
        else:
            n_samples_seen_ = np.array(n_samples_seen_)
        self.add_field(fields, "n_samples_seen_", n_samples_seen_)

        self.add_n_features(fields, model)
        self.add_feature_names(fields, model)

        return fields
