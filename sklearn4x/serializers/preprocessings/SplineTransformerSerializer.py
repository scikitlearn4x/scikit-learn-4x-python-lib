from sklearn4x.core.BaseSerializer import BaseSerializer


class SplineTransformerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_spline_transformer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        # self.add_field(fields, "_get_base_knot_positions", model._get_base_knot_positions)
        self.add_field(fields, "bsplines_", [spline.__dict__ for spline in model.bsplines_])
        self.add_field(fields, "degree", model.degree)
        self.add_field(fields, "extrapolation", model.extrapolation)
        self.add_field(fields, "include_bias", model.include_bias)
        self.add_field(fields, "knots", model.knots)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "n_features_out_", model.n_features_out_)
        self.add_field(fields, "n_knots", model.n_knots)
        self.add_field(fields, "order", model.order)

        return fields
