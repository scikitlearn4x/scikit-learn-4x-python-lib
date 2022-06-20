# from sklearn4x.core.BaseSerializer import BaseSerializer
#
#
# class KBinsDiscretizerSerializer(BaseSerializer):
#     def identifier(self):
#         return 'pp_k_bins_discretizer'
#
#     def get_fields_to_be_serialized(self, model, version):
#         fields = []
#
#         if hasattr(model, "_encoder"):
#             self.add_field(fields, "_encoder", model._encoder)
#         self.add_field(fields, "bin_edges_", [element.tolist() for element in model.bin_edges_.tolist()])
#         self.add_field(fields, "dtype", model.dtype)
#         self.add_field(fields, "encode", model.encode)
#         # self.add_field(fields, "n_bins", model.n_bins)
#         self.add_field(fields, "n_bins_", model.n_bins_)
#         self.add_field(fields, "n_features_in_", model.n_features_in_)
#         self.add_field(fields, "strategy", model.strategy)
#
#         return fields
