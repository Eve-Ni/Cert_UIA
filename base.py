import torch
import torch.nn as nn
from models import GCN, GIN
from torch_geometric.nn import global_mean_pool


def get_encoder(model_name):
    if model_name == 'gcn':
        return GCN
    # elif model_name == 'gat':
    #     return GAT
    elif model_name == 'gin':
        return GIN
    # elif model_name == 'sage':
    #     return GraphSAGE
    else:
        return None

class BaseModel(nn.Module):
    def __init__(self, inp_size, hid_size, out_size, dropout=0.0):
        super(BaseModel, self).__init__()
        Encoder = get_encoder('gcn')
        self.encoder = Encoder(inp_size, hid_size, dropout=dropout)
        self.mlp = nn.Linear(hid_size, out_size)

    def forward(self, data):
        x = self.encoder(data)
        gx = global_mean_pool(x, data.batch)
        output = self.mlp(gx)

        return output, gx