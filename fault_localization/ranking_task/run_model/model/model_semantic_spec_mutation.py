from torch import nn
import torch


class MLP(nn.Module):

	def __init__(self):
		super(MLP, self).__init__()
		
		self.mlp_all_features = nn.Linear(14, 14)

		self.output_layer = nn.Linear(14, 2)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.3)

	def forward(self, inputs):
		# spectrum is from 0 to 6
		# mutation is from 6 to 14
		spectrum = inputs[:, 0:6]
		mutation = inputs[:, 6:]
		
		all_features = torch.cat([spectrum, mutation], dim = -1)
		all_features = self.dropout(self.activation(self.mlp_all_features(all_features)))
		out = self.output_layer(all_features)
		return out
