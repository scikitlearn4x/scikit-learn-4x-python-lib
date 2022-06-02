from sklearn4x.core.BaseSerializer import BaseSerializer


class GaussianNaiveBayesSerializer(BaseSerializer):
    def identifier(self):
        return 'nb_gaussian_serializer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []
        self.add_field(fields, 'classes_', model.classes_)
        self.add_field(fields, 'class_count_', model.class_count_)
        self.add_field(fields, 'class_prior_', model.class_prior_)
        self.add_field(fields, 'theta_', model.theta_)

        if hasattr(model, 'var_'):
            self.add_field(fields, 'var_', model.var_)
        else:
            self.add_field(fields, 'var_', model.sigma_)

        if hasattr(model, 'n_features_in_'):
            self.add_field(fields, 'n_features_in_', model.n_features_in_)

        if hasattr(model, 'feature_names_in_'):
            self.add_field(fields, 'feature_names_in_', self.to_array_of_string(model.feature_names_in_))
        return fields
