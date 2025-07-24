from torch import nn
import torch


class MLP(nn.Module):

	def __init__(self, feature_type=0):
		super(MLP, self).__init__()
		self.feature_type = feature_type
		if self.feature_type == 0: # SBFL and MBFL
			self.mlp_all_features = nn.Linear(14, 14)
			self.output_layer = nn.Linear(14, 2)
		elif self.feature_type == 1: # SBFL only
			self.mlp_all_features = nn.Linear(6, 6)
			self.output_layer = nn.Linear(6, 2)
		elif self.feature_type == 2: # MBFL only
			self.mlp_all_features = nn.Linear(8, 8)
			self.output_layer = nn.Linear(8, 2)

		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.3)

	def forward(self, inputs):
		# spectrum is from 0 to 6
		# mutation is from 6 to 14
		spectrum = inputs[:, 0:6]
		mutation = inputs[:, 6:]

		inputs = []
		if self.feature_type == 0:
			inputs.append(spectrum)
			inputs.append(mutation)
		elif self.feature_type == 1:
			inputs.append(spectrum)
		elif self.feature_type == 2:
			inputs.append(mutation)

		all_features = torch.cat(inputs, dim=-1)
		all_features = self.dropout(self.activation(self.mlp_all_features(all_features)))
		out = self.output_layer(all_features)
		return out
