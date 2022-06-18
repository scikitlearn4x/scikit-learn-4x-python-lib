from sklearn4x.core.BaseSerializer import BaseSerializer


class RobustScalerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_robust_scaler'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "center_", model.center_)
        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "quantile_range", list(model.quantile_range))
        self.add_field(fields, "scale_", model.scale_)
        self.add_field(fields, "unit_variance", model.unit_variance)
        self.add_field(fields, "with_centering", model.with_centering)
        self.add_field(fields, "with_scaling", model.with_scaling)

        return fields
