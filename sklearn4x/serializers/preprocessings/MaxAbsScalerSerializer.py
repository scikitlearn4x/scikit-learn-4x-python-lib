from sklearn4x.core.BaseSerializer import BaseSerializer


class MaxAbsScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_max_abs_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "max_abs_", model.max_abs_)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "n_samples_seen_", model.n_samples_seen_)
        self.add_field(fields, "scale_", model.scale_)

        return fields
