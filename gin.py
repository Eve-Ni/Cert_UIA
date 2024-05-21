import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINConv


class GIN(nn.Module):
    def __init__(self, n_feat, n_hid, dropout):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            nn.BatchNorm1d(n_hid),
            # nn.ReLU()
            )
        nn2 = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.BatchNorm1d(n_hid),
            # nn.ReLU()
        )
        self.gc1 = GINConv(nn1, train_eps=True)
        self.gc2 = GINConv(nn2, train_eps=True)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        x = F.relu(x)
        return x