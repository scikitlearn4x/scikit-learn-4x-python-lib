from sklearn4x.core.BaseSerializer import BaseSerializer


class NormalizerSerializer(BaseSerializer):
    def identifier(self):
        return 'pp_normalizer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []

        self.add_field(fields, "copy", model.copy)
        self.add_field(fields, "n_features_in_", model.n_features_in_)
        self.add_field(fields, "norm", model.norm)

        return fields

    # ==================================================================
    # Serialize Normalizer
    #
    # Scaffolded from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
    # ==================================================================
    from sklearn4x.core.BaseSerializer import BaseSerializer

    class NormalizerSerializer(BaseSerializer):
        def identifier(self):
            return 'pp_normalizer'

        def get_fields_to_be_serialized(self, model, version):
            fields = []

            self.add_field(fields, "norm", self.get_value_or_none(model, "norm"))

            self.add_n_features(fields, model)
            self.add_feature_names(fields, model)

            return fields
