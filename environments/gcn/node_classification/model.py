import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_size=16, activation='relu', dropout=0.5):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = self.get_activation(activation)
        self.dropout = dropout

        # Ensure at least one layer is created
        num_layers = max(1, num_layers)

        # Create layers
        for i in range(num_layers):
            in_channels = num_features if i == 0 else hidden_size
            out_channels = num_classes if i == num_layers - 1 else hidden_size
            self.convs.append(GCNConv(in_channels, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation and dropout at the last layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

    def get_activation(self, activation):
        if activation == 'relu':
            return F.relu
        elif activation == 'elu':
            return F.elu
        elif activation == 'silu':
            return F.silu
        # Add more activations as needed
        else:
            raise ValueError("Unknown activation function {}".format(activation))

