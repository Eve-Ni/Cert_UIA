import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch

from sklearn.metrics import accuracy_score
import random

import os
import os.path as osp

from models import UInj, BaseModel
from data import load_dataset
from utils import save_model, load_model, train_eval_test_split, calculate_margin


from args import get_args
args = get_args("train_uinj.yaml")
f_args = get_args('pretrain_F.yaml')
base_args = get_args('train_gcn.yaml')

args.save_dir = osp.join(args.output_dir, args.data['dataset'], "checkpoints")

if not osp.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = torch.device(args.device)

# import pdb; pdb.set_trace()

def margin_loss(output, target, epsilon=0.0001):
    n_correct = len(output)
    y = output[range(n_correct), target]
    mask = torch.ones_like(output)
    mask[range(n_correct), target] = 0
    mask = mask.bool()
    y_sec = output[mask].view(n_correct, -1).max(dim=1)[0]

    # loss = torch.abs(y-y_sec-0.00001).sum() / y.shape[0]
    loss = torch.clamp(y-y_sec-epsilon, min=0).sum() / y.shape[0]
    return loss

def cls_loss(output, target):
    loss = F.cross_entropy(output, target)
    return loss 

def loss_func(output, target, deltaE):
    mask = (output.argmax(1)==target)
    l_mag = margin_loss(output[mask], target[mask])
    l_ce = cls_loss(output, target)
    l_E = F.l1_loss(deltaE, torch.zeros_like(deltaE))
    loss = 5*l_ce + 1*l_mag + 1*l_E
    return loss


def train(model, loaders):
    # Base model
    # load_model(osp.join(osp.join(base_args.output_dir, base_args.data['dataset'], "checkpoints"), base_args.test['load']), base_model)
    # base_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    best_acc = -1
    accs = []

    for epoch in range(args.training['n_epoch']):
        model.train()
        for i, data in enumerate(loaders[0]):
            data = data.to(device)
            # output_base,_ = base_model(data)
            # target = output_base.argmax(1)

            output, deltaE, _ = model(data)
            # import pdb; pdb.set_trace()
            target = data.y
            
            loss = loss_func(output, target, deltaE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f"[Train] Epoch {epoch+1}, step {i+1}, loss: {loss:.4f}")
                losses.append(loss.item())
        acc, margin = evaluate(model, loaders[1])
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
        # load_model(osp.join(osp.join(base_args.output_dir, base_args.data['dataset'], "checkpoints"), base_args.test['load']), base_model)
    
    # base_model.eval()
    model.eval()
    # base_pred = []
    pred = []
    gt = []
    margin_list = []
    inj_node = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            # output_base,_ = base_model(data)
            output, deltaE, p1 = model(data)
            # base_pred.append(output_base.argmax(1))
            pred.append(output.argmax(1))
            gt.append(data.y)
            margin = calculate_margin(output, data.y)
            margin_list.append(margin.cpu())

            for j, graph in enumerate(Batch.to_data_list(data)):
                N = graph.x.shape[0]
                inj_node += (deltaE[j][:N] > 0.5).sum().item()

            # import pdb;pdb.set_trace()
    
    # base_pred = torch.cat(base_pred)
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    # base_pred = base_pred.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()

    acc = accuracy_score(gt, pred)
    # acc = accuracy_score(gt[mask], pred[mask])

    margins = torch.cat(margin_list)
    margin = sum(margins)/len(margins)

    avg_inj_num = inj_node / len(gt)
    
    print(f'[Test] acc: {acc:.4f}, margin: {margin:.4f}, avg_inj: {avg_inj_num:.2f}')
    
    return acc, margin

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
    # Base model
    # dataset, _ = load_dataset(base_args.data['dataset'], base_args.data['path'])
    # loaders = train_eval_test_split(base_args, dataset, seed=base_args.seed)
    # base_model = BaseModel(dataset.num_node_features, base_args.model['hidden_size'], dataset.num_classes)
    # base_model.to(device)

    # UINJ model
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    losses, accs = train(model, loaders)
    evaluate(model, loaders[2], load=True)

    plot(losses, accs)


def test():
    # Base model
    # dataset, _ = load_dataset(base_args.data['dataset'], base_args.data['path'])
    # loaders = train_eval_test_split(base_args, dataset, seed=base_args.seed)
    # base_model = BaseModel(dataset.num_node_features, base_args.model['hidden_size'], dataset.num_classes)
    # base_model.to(device)

    # UINJ model
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    evaluate(model, loaders[2], load=True)


def exp4(budget):
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    dataset, _ = load_dataset(base_args.data['dataset'], base_args.data['path'])
    loaders = train_eval_test_split(base_args, dataset, seed=base_args.seed)
    base_model = BaseModel(dataset.num_node_features, base_args.model['hidden_size'], dataset.num_classes)
    base_model.to(device)

    load_model(osp.join(args.save_dir, args.test['load']), model)
    load_model(osp.join(osp.join(base_args.output_dir, base_args.data['dataset'], "checkpoints"), base_args.test['load']), base_model)
    
    model.eval()
    base_model.eval()

    rho_list = []
    pred = []
    gt = []
    pred_clean = []

    total_inj_num = budget

    with torch.no_grad():
        for i, data in enumerate(loaders[2]):
            data = data.to(device)
            output, deltaE, p1 = model(data)

            # clean acc
            output_clean, _ = base_model(data)
            pred_clean.append(output_clean.argmax(1))
            gt.append(data.y)

            mask = (output.argmax(1)==data.y)
            # rho_list.append((deltaE[mask] > 0.5).sum(dim=1))
            # rho_list += (deltaE[mask] > 0.5).sum(dim=1).tolist()

            inj_data = []
            for j, graph in enumerate(Batch.to_data_list(data)):
                N = graph.x.shape[0]
                xt = p1[j] * N
                # import pdb;pdb.set_trace()
                x_new = torch.cat([graph.x, xt.unsqueeze(0)], dim=0)

                # rho_list += (deltaE[j][:N] > 0.5).sum().tolist()
                rho_list.append((deltaE[j][:N] > 0.5).sum().item())

                # random attack
                added_edges = []
                edge_index = graph.edge_index
                if total_inj_num <= 1:
                    inj_data.append(Data(x=graph.x, edge_index=edge_index))
                    continue
                inj_num = random.randint(1, total_inj_num)
                while inj_num > N:
                    inj_num = random.randint(1, total_inj_num)
                
                indices = torch.randperm(N)
                inj_indices = indices[:inj_num].tolist()
                for i in inj_indices:
                    added_edges.append([N, i])
                    added_edges.append([i, N])
                added_edges = torch.LongTensor(added_edges).t().contiguous()
                edge_index = torch.cat([edge_index, added_edges.to(edge_index.device)], dim=1)

                inj_data.append(Data(x=x_new, edge_index=edge_index))

                total_inj_num -= inj_num
                # if budget <= 0:
                #     break
            
            new_graph_batch = Batch.from_data_list(inj_data).to(device)
            output, _ = base_model(new_graph_batch)
            pred.append(output.argmax(1))
        
        # Cert_based attack
        rho_list = torch.tensor(rho_list)
        sorted_rho_list, index = torch.sort(rho_list)
    
    pred_clean = torch.cat(pred_clean)
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    pred_clean = pred_clean.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    mr_clean = 1 - accuracy_score(gt, pred_clean)
    mr_random = 1 - accuracy_score(gt, pred)

    # import pdb;pdb.set_trace()
    rho_list = torch.tensor(rho_list)
    rho_list += 1
    sorted_rho_list, _ = torch.sort(rho_list)
    cumu_sum = torch.cumsum(sorted_rho_list, dim=0)
    max_index = torch.nonzero(cumu_sum > budget, as_tuple=False)
    # import pdb;pdb.set_trace()
    attak_success_num = max_index[0,0].item()
    mr_cert = attak_success_num / len(rho_list)
    # import pdb;pdb.set_trace()

    print(f'MR_CLEAN: {mr_clean:.4f}, MR_CERT: {mr_cert:.4f}, MR_RANDOM: {mr_random:.4f}')


def exp3(budget):
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    dataset, _ = load_dataset(base_args.data['dataset'], base_args.data['path'])
    loaders = train_eval_test_split(base_args, dataset, seed=base_args.seed)
    base_model = BaseModel(dataset.num_node_features, base_args.model['hidden_size'], dataset.num_classes)
    base_model.to(device)

    load_model(osp.join(args.save_dir, args.test['load']), model)
    load_model(osp.join(osp.join(base_args.output_dir, base_args.data['dataset'], "checkpoints"), base_args.test['load']), base_model)
    
    model.eval()
    base_model.eval()

    rho_list = []
    pred = []
    gt = []
    pred_clean = []

    total_inj_num = budget

    with torch.no_grad():
        for i, data in enumerate(loaders[2]):
            data = data.to(device)
            output, deltaE, p1 = model(data)

            # clean acc
            output_clean, _ = base_model(data)
            pred_clean.append(output_clean.argmax(1))
            gt.append(data.y)

            mask = (output.argmax(1)==data.y)
            # rho_list.append((deltaE[mask] > 0.5).sum(dim=1))
            # rho_list += (deltaE[mask] > 0.5).sum(dim=1).tolist()

            inj_data = []
            for j, graph in enumerate(Batch.to_data_list(data)):
                N = graph.x.shape[0]
                xt = p1[j] * N
                # import pdb;pdb.set_trace()
                x_new = torch.cat([graph.x, xt.unsqueeze(0)], dim=0)

                # rho_list += (deltaE[j][:N] > 0.5).sum().tolist()
                rho_list.append((deltaE[j][:N] > 0.5).sum().item())

                # random attack
                added_edges = []
                edge_index = graph.edge_index
                if total_inj_num <= 1:
                    inj_data.append(Data(x=graph.x, edge_index=edge_index))
                    continue
                inj_num = random.randint(1, total_inj_num)
                while inj_num > N:
                    inj_num = random.randint(1, total_inj_num)
                
                indices = torch.randperm(N)
                inj_indices = indices[:inj_num].tolist()
                for i in inj_indices:
                    added_edges.append([N, i])
                    added_edges.append([i, N])
                added_edges = torch.LongTensor(added_edges).t().contiguous()
                edge_index = torch.cat([edge_index, added_edges.to(edge_index.device)], dim=1)

                inj_data.append(Data(x=x_new, edge_index=edge_index))

                total_inj_num -= inj_num
                # if budget <= 0:
                #     break
            
            new_graph_batch = Batch.from_data_list(inj_data).to(device)
            output, _ = base_model(new_graph_batch)
            pred.append(output.argmax(1))
    
    pred_clean = torch.cat(pred_clean)
    pred = torch.cat(pred)
    gt = torch.cat(gt)
    pred_clean = pred_clean.data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    mr_clean = 1 - accuracy_score(gt, pred_clean)
    mr_random = 1 - accuracy_score(gt, pred)

    # import pdb;pdb.set_trace()
    rho_list = torch.tensor(rho_list)
    rho_list += 6
    sorted_rho_list, _ = torch.sort(rho_list)
    cumu_sum = torch.cumsum(sorted_rho_list, dim=0)
    max_index = torch.nonzero(cumu_sum > budget, as_tuple=False)
    # import pdb;pdb.set_trace()
    attak_success_num = max_index[0,0].item()
    mr_cert = attak_success_num / len(rho_list)
    # import pdb;pdb.set_trace()

    print(f'MR_CLEAN: {mr_clean:.4f}, MR_CERT: {mr_cert:.4f}, MR_RANDOM: {mr_random:.4f}')
    # return asr_cert, asr_random


def exp2():
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    load_model(osp.join(args.save_dir, args.test['load']), model)
    model.eval()

    rho_list = []
    perturb_data = []
    pred = []
    gt = []

    with torch.no_grad():
        for i, data in enumerate(loaders[2]):
            data = data.to(device)
            output, deltaE, p1 = model(data)

            pred.append(output.argmax(1))
            gt.append(data.y)

            # rho_list += (deltaE > 0.5).sum(dim=1).tolist()
        
            # generate perturbed data 
            for j, graph in enumerate(Batch.to_data_list(data)):
                N = graph.x.shape[0]
                xt = p1[j] * N

                rho_list.append((deltaE[j][:N] > 0.5).sum().item())
                # import pdb;pdb.set_trace()
                # x_new = torch.cat([graph.x, xt.unsqueeze(0)], dim=0).sum(dim=1).unsqueeze(1)
                x_new = torch.cat([graph.x, xt.unsqueeze(0)], dim=0)
                # import pdb;pdb.set_trace()
                perturb_data.append(Data(x=x_new, edge_index=graph.edge_index, y=graph.y))
                
    # save perturb data
    file_path = "./output/perturb_data/"+args.data['dataset']+".pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(perturb_data, file_path)

    pred = torch.cat(pred)
    gt = torch.cat(gt)
    mask = (pred==gt)
    # import pdb;pdb.set_trace()
    rho_list = torch.tensor(rho_list)
    rho_list = rho_list[mask]
    acc_0 = len(rho_list) / len(pred)
    sort_rho,_ = torch.sort(rho_list)
    
    print(f'rho=0, acc: {acc_0:.4f}')
    print(f'The perturbation radius list: {sort_rho}')


def exp1():
    dataset, maxN = load_dataset(args.data['dataset'], args.data['path'])
    loaders = train_eval_test_split(args, dataset, seed=args.seed)
    model = UInj(maxN, dataset.num_node_features, dataset.num_classes, args, f_args, base_args)
    model.to(device)

    dataset, _ = load_dataset(base_args.data['dataset'], base_args.data['path'])
    loaders = train_eval_test_split(base_args, dataset, seed=base_args.seed)
    base_model = BaseModel(dataset.num_node_features, base_args.model['hidden_size'], dataset.num_classes)
    base_model.to(device)

    load_model(osp.join(args.save_dir, args.test['load']), model)
    load_model(osp.join(osp.join(base_args.output_dir, base_args.data['dataset'], "checkpoints"), base_args.test['load']), base_model)
    
    model.eval()
    base_model.eval()

    pred_old = []
    gt = []
    pred_new = []

    with torch.no_grad():
        for i, data in enumerate(loaders[2]):
            data = data.to(device)
            output, deltaE, p1 = model(data)

            pred_old.append(output.argmax(1))
            gt.append(data.y)

            deltaE_bin = torch.where(deltaE > 0.5, torch.ones_like(deltaE), torch.zeros_like(deltaE))

            new_data = []
            for j, graph in enumerate(Batch.to_data_list(data)):
                N = graph.x.shape[0]
                xt = p1[j] * N
                x_new = torch.cat([graph.x, xt.unsqueeze(0)], dim=0)

                edge_index = graph.edge_index
                added_edges = []
                row_vector = deltaE_bin[j]
                zero_indices = torch.nonzero(row_vector[:N] == 0).squeeze(1)
                # import pdb;pdb.set_trace()
                if zero_indices.size(0)==0:
                    new_data.append(Data(x=x_new, edge_index=edge_index))
                    continue
                rand_index = torch.randint(high=zero_indices.size(0), size=(1,)).item()
                index = zero_indices[rand_index].item()
                added_edges.append([N, index])
                added_edges.append([index, N])
                added_edges = torch.LongTensor(added_edges).t().contiguous()

                edge_index = torch.cat([edge_index, added_edges.to(edge_index.device)], dim=1)

                new_data.append(Data(x=x_new, edge_index=edge_index))
            
            new_graph_batch = Batch.from_data_list(new_data).to(device)
            output_new, _ = base_model(new_graph_batch)
            pred_new.append(output_new.argmax(1))
    
    pred_old = torch.cat(pred_old)
    pred_new = torch.cat(pred_new)
    gt = torch.cat(gt)
    mask1 = (pred_old==gt)
    # mask2 = (pred_new[mask1]==gt[mask1])
    # mask2 = (pred_new==gt)
    mask2 = (pred_new==pred_old)
    acc_before = torch.sum(mask1).item()/len(mask1)
    acc_after = torch.sum(mask2).item()/len(mask2)

    print(f'Robust_acc: {acc_before:.4f}, flip_rate: {1-acc_after:.4f}')

                


