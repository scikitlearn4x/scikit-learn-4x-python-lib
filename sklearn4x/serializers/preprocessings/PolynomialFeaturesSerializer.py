# from sklearn4x.core.BaseSerializer import BaseSerializer
#
#
# class PolynomialFeaturesSerializer(BaseSerializer):
#     def identifier(self):
#         return 'pp_polynomial_features'
#
#     def get_fields_to_be_serialized(self, model, version):
#         fields = []
#
#         # self.add_field(fields, "_combinations", model._combinations)
#         # self.add_field(fields, "_num_combinations", model._num_combinations)
#         self.add_field(fields, "_max_degree", model._max_degree)
#         self.add_field(fields, "_min_degree", model._min_degree)
#         self.add_field(fields, "_n_out_full", model._n_out_full)
#         self.add_field(fields, "degree", model.degree)
#         self.add_field(fields, "include_bias", model.include_bias)
#         self.add_field(fields, "interaction_only", model.interaction_only)
#         self.add_field(fields, "n_features_in_", model.n_features_in_)
#         self.add_field(fields, "n_input_features_", model.n_input_features_)
#         self.add_field(fields, "n_output_features_", model.n_output_features_)
#         self.add_field(fields, "order", model.order)
#         self.add_field(fields, "powers_", model.powers_)
#
#         return fields
