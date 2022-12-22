# import the pytorch library into environment and check its version
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader
import numpy as np
torch.cuda.is_available()
import numpy as np
import matplotlib.mlab as mlab
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch import Conv1d, Conv2d, Linear, Dropout
import torch.nn.functional as F


class GCN_TC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        self.onedcnn = Conv1d(12, 12, 10, stride=2)
        self.twodcnn = Conv2d(1,1,(1,10),stride=(1,2))
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels, out_channels)
        self.linear2 = Linear(out_channels, 1)
        self.dropout = Dropout(p=0.2)
        self.relu = F.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, edge_index, batch):
        x = torch.unsqueeze(x,0)
        x2 = self.twodcnn(x)
        x3 = self.twodcnn(x2)
        x4 = self.twodcnn(x3)
        x4 = torch.squeeze(x4)
        output1 = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x4, edge_index)), edge_index)), edge_index)
        output2 = global_mean_pool(output1, batch)
        output = self.sigmoid(self.linear2(self.relu(self.linear1(output2))))

        return output