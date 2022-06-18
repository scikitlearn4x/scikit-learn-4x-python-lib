from sklearn4x.core.BaseSerializer import BaseSerializer


class MinMaxScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_min_max_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "clip", model.clip)
        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "data_max_", model.data_max_)
        self.add_field(fields, "data_min_", model.data_min_)
        self.add_field(fields, "data_range_", model.data_range_)
        self.add_field(fields, "feature_range", model.feature_range)
        self.add_field(fields, "min_", model.min_)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "n_samples_seen_", model.n_samples_seen_)
        self.add_field(fields, "scale_", model.scale_)

        return fields