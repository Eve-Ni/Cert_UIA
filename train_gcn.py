import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from sklearn.metrics import accuracy_score

import os
import os.path as osp

from models import BaseModel
from data import load_dataset
from utils import save_model, load_model, train_eval_test_split

from args import get_args
args = get_args("train_gcn.yaml")

args.save_dir = osp.join(args.output_dir, args.data['dataset'], "checkpoints")

if not osp.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = torch.device(args.device)

def loss_func(output, target):
    loss = F.cross_entropy(output, target)
    return loss

def train(model, loaders):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    best_acc = -1
    accs = []
    for epoch in range(args.training['n_epoch']):
        model.train()
        for i, data in enumerate(loaders[0]):
            data = data.to(device)
            output,_ = model(data)
            # import pdb; pdb.set_trace()
            loss = loss_func(output, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f"[Train] Epoch {epoch+1}, step {i+1}, loss: {loss:.4f}")
                losses.append(loss.item())
        acc = evaluate(model, loaders[1])
        print(f'[Evaluate] Epoch {epoch+1}, acc:{acc:.4f}')
        if acc > best_acc+1e-12:
            best_acc = acc
            save_model(model, osp.join(args.save_dir, "best.pt"))
        accs.append(acc)
    save_model(model, osp.join(args.save_dir, "model.pt"))
    return losses, accs


def evaluate(model, data_loader, load=False):
    if load:
        print("load", args.test['load'])
        load_model(osp.join(args.save_dir, args.test['load']), model)
    model.eval()
    pred =[]
    gt = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            output,_ = model(data)
            pred.append(output.argmax(1))
            gt.append(data.y)
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    pred = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    acc = accuracy_score(gt, pred)
    print(f'[Test] acc: {acc:.4f}')
    return acc

def plot(losses, accs):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.plot(losses)
    plt.xlabel('steps')
    plt.title('loss')
    plt.subplot(122)
    plt.plot(accs)
    plt.title('acc')
    plt.xlabel('epoch')
    plt.savefig(osp.join(osp.dirname(args.save_dir), 'training.png'),dpi=200,bbox_inches='tight',pad_inches=0.1)
    plt.show()


def run():
    dataset, _ = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset,seed=args.seed)
    model = BaseModel(dataset.num_node_features, args.model['hidden_size'], dataset.num_classes)
    model.to(device)

    losses, accs = train(model, loaders)
    evaluate(model, loaders[2], load=True)

    plot(losses, accs)


def test():
    dataset, _ = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = BaseModel(dataset.num_node_features, args.model['hidden_size'], dataset.num_classes)
    model.to(device)

    evaluate(model, loaders[2], load=True)