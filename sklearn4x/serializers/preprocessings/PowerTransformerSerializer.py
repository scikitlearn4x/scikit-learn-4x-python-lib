from sklearn4x.core.BaseSerializer import BaseSerializer


class PowerTransformerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_power_transformer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        if hasattr(model, "_scaler"):
            self.add_field(fields, "_scaler", model._scaler)
        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "lambdas_", model.lambdas_)
        self.add_field(fields, "method", model.method)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "standardize", model.standardize)

        return fields
