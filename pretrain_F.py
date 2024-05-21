import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from sklearn.metrics import accuracy_score

import os
import os.path as osp

from models import FModel
from data import load_dataset
from utils import save_model, load_model, train_eval_test_split

from args import get_args
args = get_args("pretrain_F.yaml")
base_args = get_args("train_gcn.yaml")

args.save_dir = osp.join(args.output_dir, args.data['dataset'], "checkpoints")

if not osp.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = torch.device(args.device)

# def loss_func(output, target):
#     loss = F.nll_loss(output, target)
#     return loss
# def l1_loss(output):
#     return F.l1_loss(output, torch.zeros_like(output))

# def deltaH_gcn1(scale, p, w):
#     # scale: (B,)
#     # p: (B,d)
#     # w: (d,d)
#     kpw = scale.unsqueeze(1) * torch.matmul(p, w)
#     return kpw


def train(model, loaders):
    optimizer = torch.optim.Adam(model.sampler.parameters(), lr=args.training['lr_p'])
    losses = []
    for epoch in range(args.training['n_epoch']):
        model.train()
        for i, data in enumerate(loaders[0]):
            data = data.to(device)
            _, loss = model(data)
            # freeze F
            # for p in model.F.parameters():
            #     p.requires_grad = False
            
            # deltaH =  #Kpw
            # deltaE = model.F(deltaH)
            # loss = l1_loss(deltaE)

            for p in model.sampler.parameters():
                p.grad = None
            loss.backward()
            optimizer.step()

            # for p in model.F.parameters():
            #     p.requires_grad = True
            if (i+1) % 10 == 0:
                print(f"[Train] Epoch {epoch+1}, step {i+1}, loss: {loss:.4f}")
                losses.append(loss.item())

    save_model(model, osp.join(args.save_dir, "F_model.pt"))
    return losses

def plot(losses):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3))
    plt.subplot(111)
    plt.plot(losses)
    plt.xlabel('steps')
    plt.title('loss')
    plt.savefig(osp.join(osp.dirname(args.save_dir), 'training.png'),dpi=200,bbox_inches='tight',pad_inches=0.1)
    plt.show()

def run():
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    print("maxN:", maxN)
    loaders = train_eval_test_split(args, dataset,seed=args.seed)
    model = FModel(maxN, dataset.num_node_features, dataset.num_classes, args, base_args)
    model.to(device)

    losses = train(model, loaders)
    plot(losses)