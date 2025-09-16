from torch import nn
import torch


class MLP(nn.Module):

	def __init__(self, feature_type=0):
		super(MLP, self).__init__()
		self.feature_type = feature_type
		if self.feature_type == 0: # only SBFL
			self.mlp_all_features = nn.Linear(6, 6)
			self.output_layer = nn.Linear(6, 2)
		elif self.feature_type == 1: # only MBFL
			self.mlp_all_features = nn.Linear(2, 2)
			self.output_layer = nn.Linear(2, 2)
		elif self.feature_type == 2: # only ST
			self.mlp_all_features = nn.Linear(1, 1)
			self.output_layer = nn.Linear(1, 2)
		elif self.feature_type == 3: # No SBFL
			self.mlp_all_features = nn.Linear(3, 3)
			self.output_layer = nn.Linear(3, 2)
		elif self.feature_type == 4: # No MBFL
			self.mlp_all_features = nn.Linear(7, 7)
			self.output_layer = nn.Linear(7, 2)
		elif self.feature_type == 5: # No ST
			self.mlp_all_features = nn.Linear(8, 8)
			self.output_layer = nn.Linear(8, 2)
		elif self.feature_type == 6: # All features
			self.mlp_all_features = nn.Linear(9, 9)  # 6(SBFL) + 2(MBFL) + 1(ST) = 9 features
			self.output_layer = nn.Linear(9, 2)

		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.3)

	def forward(self, inputs):
		# spectrum is from 0 to 6
		# mutation is from 6 to 8
		# ST is from 8 to 9
		# st_linear is from 9 to end
		spectrum = inputs[:, 0:6]
		mutation = inputs[:, 6:8]
		st = inputs[:, 8:9]
		st_linear = inputs[:, 9:]

		if self.feature_type == 0:
			inputs.append(spectrum)
		elif self.feature_type == 1:
			inputs.append(mutation)
		elif self.feature_type == 2:
			inputs.append(st)
		elif self.feature_type == 3:
			inputs.append(spectrum)
			inputs.append(mutation)
		elif self.feature_type == 4:
			inputs.append(spectrum)
			inputs.append(st)
		elif self.feature_type == 5:
			inputs.append(mutation)
			inputs.append(st)
		elif self.feature_type == 6:
			inputs.append(spectrum)
			inputs.append(mutation)
			inputs.append(st)

		all_features = torch.cat(inputs, dim=-1)
		all_features = self.dropout(self.activation(self.mlp_all_features(all_features)))
		out = self.output_layer(all_features)

		return out
