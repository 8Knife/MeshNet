import copy
import os
import time
import random
import torch
from torch.autograd import Variable

import torch_npu
from torch_npu.contrib import transfer_to_npu


import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import math
import numpy as np
from config import get_train_config
from data import ModelNet40
from models import MeshNet
from utils.retrival import append_feature, calculate_map


cfg = get_train_config()
# os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
# os.environ['ASCEND_RT_VISIBLE_DEVICES'] = cfg['npu_devices']

# seed
seed = cfg['seed']
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

# dataset
data_set = {
    x: ModelNet40(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=cfg['batch_size'], num_workers=4, shuffle=True, pin_memory=False)
    for x in ['train', 'test']
}


def train_model(model, criterion, optimizer, scheduler, cfg, start_epoch=1):

    best_acc = 0.0
    best_map = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, cfg['max_epoch'] + 1):

        epoch_start_time = time.time()
        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        # adjust_learning_rate(cfg, epoch, optimizer)
        for phrase in ['train', 'test']:

            if phrase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            ft_all, lbl_all = None, None

            for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader[phrase]):
                centers = centers.to(device)
                corners = corners.to(device)
                normals = normals.to(device)
                neighbor_index = neighbor_index.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(phrase == 'train'):
                    outputs, feas = model(centers, corners, normals, neighbor_index)

                    if not torch.isfinite(outputs).all():
                        print("Output NaN")
                        optimizer.zero_grad()
                        continue

                    outputs = torch.clamp(outputs, -50, 50)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, targets)

                    if phrase == 'train':
                        optimizer.zero_grad()
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        optimizer.step()

                    if phrase == 'test' and cfg['retrieval_on']:
                        ft_all = append_feature(ft_all, feas.detach().cpu())
                        lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)

                    running_loss += loss.item() * centers.size(0)
                    running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / len(data_set[phrase])
            epoch_acc = running_corrects.double() / len(data_set[phrase])

            if phrase == 'train':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))
                scheduler.step()

            if phrase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "best_map": best_map
                    }, os.path.join(cfg['ckpt_root'], 'MeshNet_best.pkl'))

                print_info = '{} Loss: {:.4f} Acc: {:.4f} (best {:.4f})'.format(phrase, epoch_loss, epoch_acc, best_acc)

                if cfg['retrieval_on']:
                    epoch_map = calculate_map(ft_all, lbl_all)
                    if epoch_map > best_map:
                        best_map = epoch_map
                    print_info += ' mAP: {:.4f}'.format(epoch_map)

                if epoch % cfg['save_steps'] == 0:
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "best_map": best_map
                    }, os.path.join(cfg['ckpt_root'], f"{epoch}.pkl"))

                epoch_time = time.time() - epoch_start_time
                print(f"Epoch: {epoch}\n Epoch time: {epoch_time:.2f}s\n")
                print(print_info)

    print('Best val acc: {:.4f}'.format(best_acc))
    print('Config: {}'.format(cfg))

    return best_model_wts


if __name__ == '__main__':

    # prepare model
    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    device = torch.device("npu:7")

    model = model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # scheduler
    if cfg['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_epoch'])

    # start training
    # start training
    if not os.path.exists(cfg['ckpt_root']):
        os.mkdir(cfg['ckpt_root'])

    checkpoint_path = os.path.join(cfg['ckpt_root'], "checkpoint.pkl")

    start_epoch = 1

    pkl_files = [f for f in os.listdir(cfg['ckpt_root']) if f.endswith('.pkl') and f[0].isdigit()]

    if len(pkl_files) > 0:
        epochs = [int(f.split('.')[0]) for f in pkl_files]

        last_epoch = max(epochs)

        last_ckpt = os.path.join(cfg['ckpt_root'], f"{last_epoch}.pkl")

        print("Resume from:", last_ckpt)

        checkpoint = torch.load(last_ckpt)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1

        print("Resume epoch:", start_epoch)

    best_model_wts = train_model(model, criterion, optimizer, scheduler, cfg, start_epoch)
