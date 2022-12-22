from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels, out_channels)
        self.linear2 = Linear(out_channels, 1)
        self.relu = F.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, edge_index, batch):

        output1 = self.conv2(self.relu(self.conv1(x, edge_index)), edge_index)
        output2 = global_mean_pool(output1, batch)

        output = self.sigmoid(self.linear2(self.relu(self.linear1(output2))))
        return output