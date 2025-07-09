
import torch
import warnings
# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=UserWarning, message="The torch.cuda.*DtypeTensor constructors are no longer recommended")


import os, sys
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
import math
from sklearn import metrics
from random import shuffle

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm
import itertools

''' Own library '''
own_dir = os.path.join('../')
sys.path.append(own_dir)

from config import *
from utils.init_model import *
from utils.validation import *
from utils.style_transfer import *
from utils.style_stats import *
from models.frmil import FeatMag

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
class LocalUpdate:
    def __init__(self, args, local_epochs=20, bag_train=None, bag_test=None, center_id=None, local_milnet=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.bag_test = bag_test
        self.bag_train = pd.concat([bag_train], axis=0)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.local_epochs = local_epochs
        self.center_id = center_id
        self.local_best_score = 0
        self.local_milnet = local_milnet

        self.style_stats = StyleStatistics()
        self.style_transfer = StyleTransfer(args=args)

    
    def extract_statistics(self, args, df, verbose=True):
        stats = {'bag_mu': [], 'bag_var': []}
    
        bags = df.values.tolist()
    
        for i, item in enumerate(bags):
            file_name = os.path.basename(item[0]).replace('csv', 'pt')
    
            # Determine path to the .pt file
            if 'c17' in args.dataset:
                bag_item = os.path.join(C17_PATH, 'pt', args.backbone, file_name)
            elif 'tcga_rcc' in args.dataset:
                bag_item = os.path.join(TCGA_RCC_PATH, 'pt', args.backbone, file_name)
            elif 'her2' in args.dataset:
                base = (
                    TCGA_HEROHE_PATH if 'herohe' in item[0]
                    else TCGA_BRCA_PATH if 'brca' in item[0]
                    else TCGA_YALE_PATH
                )
                bag_item = os.path.join(base, 'pt', args.backbone, file_name)
            else:
                continue
    
            # Get the corresponding style path
            style_path = bag_item.replace('/pt/', '/style_pt/')
    
            if not os.path.exists(style_path):
                continue  # Skip if no style file exists
    
            # Load precomputed stats
            style = torch.load(style_path, weights_only=False)

            for mu, var in zip(style['centroids_means'], style['centroids_stds']):
                if not (math.isnan(mu) or math.isnan(var)):
                    stats['bag_mu'].append(torch.tensor(mu).view(1, -1))
                    stats['bag_var'].append(torch.tensor(var).view(1, -1))
    
            if verbose:
                sys.stdout.write(f'\rLoading precomputed stats [{i+1}/{len(bags)}]')
    
        print()
        return stats

    def train(self, args, milnet, cached_statistics):
        self.local_best_score = 0
        center_id = self.center_id

        label_counts = self.bag_train['label'].value_counts().sort_index()
        total_samples = len(self.bag_train)
        class_weights = (total_samples / (len(label_counts) * label_counts)).values
        class_weights = class_weights / class_weights.sum()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        if args.federated == 'FedAvg':
            self.local_milnet = copy.deepcopy(milnet)
        elif args.federated == 'FedBN':
            bn_exclude = [k for k in milnet.state_dict() if not any(t in k for t in ["running_mean", "running_var", "num_batches_tracked"])]
            filtered_state = {k: milnet.state_dict()[k] for k in bn_exclude}
            self.local_milnet.load_state_dict(OrderedDict(filtered_state), strict=False)

        best_model = copy.deepcopy(self.local_milnet)

        criterion = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(self.local_milnet.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.local_epochs, 5e-6)

        tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.local_milnet.train()

        epoch_loss = []
        bags = self.bag_train.values.tolist()

        for epoch in range(1, self.local_epochs + 1):
            total_loss = 0
            for i, item in enumerate(bags):
                file_name = os.path.basename(item[0]).replace('csv', 'pt')
                if 'c17' in args.dataset:
                    path = os.path.join(C17_PATH, 'pt', args.backbone, file_name)
                elif 'tcga_rcc' in args.dataset:
                    root = os.path.join(TCGA_RCC_PATH, 'pt', args.backbone)
                    path = os.path.join(root, file_name.replace('.csv', 'pt'))
                elif 'her2' in args.dataset:
                    base = TCGA_HEROHE_PATH if 'herohe' in item[0] else TCGA_BRCA_PATH if 'brca' in item[0] else TCGA_YALE_PATH
                    path = os.path.join(base, 'pt', args.backbone, file_name)
                else:
                    continue

                if not os.path.isfile(path):
                    print('Not found')
                    continue

                optimizer.zero_grad()
                data = torch.load(path, map_location=device)
                label = int(item[1])

                if args.num_classes != 1:
                    label_onehot = np.zeros(args.num_classes)
                    label_onehot[label] = 1
                    bag_label = Variable(torch.FloatTensor([label_onehot]).cuda())
                else:
                    bag_label = torch.tensor([label], dtype=torch.float32).to(device)

                feats = tensor_type(data[:, :args.feats_size])
                feats = feats[torch.randperm(len(feats))]
                feats = dropout_patches(feats, 1 - args.dropout_patch).view(-1, args.feats_size)

                # random_idx = torch.randint(0, feats.shape[0], (1,)).item()
                bag_mu, bag_var = self.style_stats.get_statistics(feats)

                if np.random.rand() < 0.5:
                    feats = self.style_transfer(feats, cached_statistics)
                
                if args.model == 'HistoFL':
                    bag_pred, _, _, _ = self.local_milnet(feats)
                    loss = criterion(bag_pred.view(1, -1), bag_label.view(1, -1))
                elif args.model == 'frmil':
                    bag_pred, _, max_c = self.local_milnet(feats)
                    max_c = torch.max(max_c, 1)[0].expand_as(bag_label)
                    loss = 0.5 * F.binary_cross_entropy(max_c, bag_label.float()) + \
                           0.5 * F.cross_entropy(bag_pred, bag_label)
                else:
                    continue

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                sys.stdout.write(f'\rExp: {args.experiment} Institute: {center_id} Epoch {epoch} Bag [{i+1}/{len(bags)}] Loss: {loss.item():.4f}')

            epoch_loss.append(total_loss / len(bags))
            scheduler.step()

            # val_loss, avg_score, aucs, _, stats = test(args, self.bag_test, self.local_milnet, criterion)
            # if get_current_score(avg_score, aucs) > self.local_best_score:
        best_model = copy.deepcopy(self.local_milnet)
        # self.local_best_score = get_current_score(avg_score, aucs)
        # print(f'\nSave Best Model !! Score: {self.local_best_score:.4f}')
        
        self.lr = scheduler.get_last_lr()[0]
        return best_model, sum(epoch_loss) / self.local_epochs, len(bags)

