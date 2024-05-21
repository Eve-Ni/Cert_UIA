import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from .F_model import Sampler, FModel
from utils import load_model


class UInj(nn.Module):
    def __init__(self, maxN, inp_size, out_size, args, f_args, base_args):
        super(UInj, self).__init__()
        
        self.sampler = Sampler(base_args.model['hidden_size'], args.sampler['hidden_size'], inp_size)

        self.f_model = FModel(maxN, inp_size, out_size, f_args, base_args)
        load_model(args.load, self.f_model)

        for p in self.f_model.base.parameters():
            p.requires_grad = False
        
        for p in self.f_model.F.parameters():
            p.requires_grad = False

    def _addp(self, p1, p2, data):
        re_graph_list = []
        for i, graph in enumerate(Batch.to_data_list(data)):
            x = graph.x + p1[i] + p2[i]
            re_graph_list.append(Data(x=x,edge_index=graph.edge_index,y=graph.y))
        return Batch.from_data_list(re_graph_list)
    
    def _deltaH(self, p, data):
        p_data = []
        for i, graph in enumerate(Batch.to_data_list(data)):
            N = graph.x.shape[0]
            K = torch.ones(N,1).to(p.device)
            Kp = torch.matmul(K, p[i].unsqueeze(0))
            p_data.append(Data(x=Kp, edge_index=graph.edge_index))
        p_data = Batch.from_data_list(p_data)
        _, Kpw = self.f_model.base(p_data)
        return Kpw
    
    def forward(self, data):
        _, gx = self.f_model.base(data)
        p1, _ = self.f_model.sampler(gx)
        p2, _ = self.sampler(gx)
        new_data = self._addp(p1, p2, data)
        output, gx = self.f_model.base(new_data)

        deltaH = self._deltaH(p2, data) #Kpw
        deltaE = self.f_model.F(deltaH)

        return output, deltaE, p1