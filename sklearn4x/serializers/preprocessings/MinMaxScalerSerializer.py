from sklearn4x.core.BaseSerializer import BaseSerializer


class MinMaxScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_min_max_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []


        return fields

    # ==================================================================
    # Serialize MinMaxScaler
    #
    # Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    # ==================================================================
    from sklearn4x.core.BaseSerializer import BaseSerializer

    class MinMaxScalerSerializer(BaseSerializer):
        def identifier(self):
            return 'pp_min_max_scaler'

        def get_fields_to_be_serialized(self, model, version):
            fields = []

            self.add_field(fields, "min_", self.get_value_or_none(model, "min_"))
            self.add_field(fields, "scale_", self.get_value_or_none(model, "scale_"), version=version, min_version='0.17')
            self.add_field(fields, "data_min_", self.get_value_or_none(model, "data_min_"), version=version, min_version='0.17')
            self.add_field(fields, "data_max_", self.get_value_or_none(model, "data_max_"), version=version, min_version='0.17')
            self.add_field(fields, "data_range_", self.get_value_or_none(model, "data_range_"), version=version, min_version='0.17')
            self.add_field(fields, "n_samples_seen_", self.get_value_or_none(model, "n_samples_seen_"))
            self.add_field(fields, "feature_range", self.get_value_or_none(model, "feature_range"))
            self.add_field(fields, "clip", model.clip)

            self.add_n_features(fields, model)
            self.add_feature_names(fields, model)

            return fields