from sklearn4x.core.BaseSerializer import BaseSerializer


class StandardScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_standard_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "mean_", model.mean_)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "n_samples_seen_", model.n_samples_seen_)
        self.add_field(fields, "scale_", model.scale_)
        self.add_field(fields, "var_", model.var_)
        self.add_field(fields, "with_mean", model.with_mean)
        self.add_field(fields, "with_std", model.with_std)

        return fields
