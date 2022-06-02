from sklearn4x.core.BaseSerializer import BaseSerializer


class CategoricalNaiveBayesSerializer(BaseSerializer):
    def identifier(self):
        return 'nb_categorical_serializer'

    def get_fields_to_be_serialized(self, model, version):
        fields = []
        self.add_field(fields, 'category_count_', model.category_count_)
        self.add_field(fields, 'class_count_', model.class_count_)
        self.add_field(fields, 'class_log_prior_', model.class_log_prior_)
        self.add_field(fields, 'classes_', model.classes_)
        self.add_field(fields, 'feature_log_prob_', model.feature_log_prob_)
        self.add_field(fields, 'n_features_', model.n_features_, version, max_version='1.2')
        self.add_field(fields, 'n_features_in_', model.n_features_in_, version, min_version='0.24')
        self.add_field(fields, 'n_categories_', model.n_categories_, version, min_version='0.24')

        if hasattr(model, 'feature_names_in_'):
            self.add_field(fields, 'feature_names_in_', self.to_array_of_string(model.feature_names_in_))
        return fields