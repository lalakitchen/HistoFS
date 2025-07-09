import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm

# Local library
sys.path.append('../')
from config import *
from Federated.FedAvg import FedWeightAvg, FedWeightAvgBatchNorm, FedProx
from Federated.LocalUpdate import *
from utils.init_model import *
from utils.validation import *




def main():
    parser = argparse.ArgumentParser(description='Train F-MIL')

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(5,))
    parser.add_argument('--weight_decay', default=5e-3, type=float)
    parser.add_argument('--dataset', default='c17', type=str)
    parser.add_argument('--model', default='HistoFL', type=str)
    parser.add_argument('--dropout_patch', default=0, type=float)
    parser.add_argument('--dropout_node', default=0, type=float)
    parser.add_argument('--non_linearity', default=1, type=float)
    parser.add_argument('--average', type=bool, default=False)
    parser.add_argument('--eval_scheme', default='5-fold-cv', type=str)
    parser.add_argument('--experiment', default=None, type=str, help='Optional experiment name override')

    parser.add_argument("--backbone", default='dino', type=str, choices=['resnet', 'dino'])
    parser.add_argument("--style", default='Our', type=str)
    parser.add_argument("--federated", default='FedAvg', type=str, choices=['FedAvg', 'FedBN', 'FedProx'])
    parser.add_argument("--round_fl", default=40, type=int)
    parser.add_argument("--leave_out", default=0, type=int)
    parser.add_argument("--auth_module", default=True, type=bool, choices=[False, True])

    args = parser.parse_args()
    args.feats_size = {'resnet': 1024, 'dino': 384}[args.backbone]
    args.num_classes = {'her2': 2, 'tcga_rcc': 3, 'c17': 2}[args.dataset]
    args.batch_norm = {'FedAvg': False, 'FedBN': True, 'FedProx': False}[args.federated]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_index)
    args.device = torch.device('cuda:{}'.format(args.gpu_index[0]))

    if args.model == 'HistoFL':
        import models.histofl as mil
    elif args.model == 'frmil':
        import models.frmil as mil

    print('FEDERATED LEARNING')

    df_train = pd.read_csv(os.path.join('..', 'labels', args.dataset, f'train_seen_leave_{args.leave_out}.csv'))
    df_test = pd.read_csv(os.path.join('..', 'labels', args.dataset, f'test_seen_leave_{args.leave_out}.csv'))
    df_train_grouped = df_train.groupby('center')

    distinct_centers = df_train['center'].unique()

    if args.experiment is None:
        args.experiment = f"{args.backbone}_{args.federated if args.federated in ('FedBN', 'FedProx') else args.style}_leave_out_{args.leave_out}"
        if args.auth_module:
            args.experiment += '_Auth'

    print(f'Dataset: {args.dataset}, Experiment: {args.experiment}')
    save_path = os.path.join('..', 'weightsX', args.model, args.dataset, args.experiment)
    os.makedirs(save_path, exist_ok=True)

    milnet, criterion, optimizer, scheduler = init_model(mil, args)
    milnet.train()
    clients = {}
    cached_statistics = {}

    # Step 1: Initialize clients and extract per-center statistics
    for center, group in df_train_grouped:
        client = LocalUpdate(
            args,
            bag_train=group,
            bag_test=df_test,
            center_id=center,
            local_milnet=copy.deepcopy(milnet).cuda()
        )
        clients[center] = client
        cached_statistics[center] = client.extract_statistics(args, group)
        print(f'Center: {center}, Train shape: {group.shape}, Test shape: {df_test.shape}')

    # Step 2: Precompute combined statistics from all *other* centers
    other_center_stats = {}
    for center in cached_statistics:
        mu_all, var_all = [], []
        for other_center in cached_statistics:
            if other_center == center:
                continue
            mu_all.extend(cached_statistics[other_center]['bag_mu'])
            var_all.extend(cached_statistics[other_center]['bag_var'])
        other_center_stats[center] = {'bag_mu': mu_all, 'bag_var': var_all}
    
    # Step 3: Training loop
    fold_best_score = 0
    best_ac, best_auc = 0, 0
    milnet, criterion, optimizer, scheduler = init_model(mil, args)
    milnet.train()
    
    for round_com in range(args.round_fl):
        print(f"\n Round {round_com}: ")
        w_locals, losses, sizes = [], [], []
    
        for center in distinct_centers:
            # Upload stats from other centers (excluding this one)
            w_local, loss, size = clients[center].train(
                args,
                milnet=copy.deepcopy(milnet).cuda(),
                cached_statistics=other_center_stats[center]
            )
            w_locals.append(w_local)
            losses.append(loss)
            sizes.append(size)
    
        # Optional: track previous round statistics
    
        # Federated aggregation
        if args.federated == 'FedAvg':
            w_glob = FedWeightAvg(milnet, w_locals)
        elif args.federated == 'FedBN':
            w_glob = FedWeightAvgBatchNorm(milnet, w_locals)
        elif args.federated == 'FedProx':
            w_glob = FedProx(milnet, w_locals)
    
        # Load global weights into model
        milnet.load_state_dict(w_glob)
    
        # Evaluate the updated global model
        print('Evaluate All Centers:')
        val_loss, avg_score, aucs, thresholds_optimal = test(args, df_test, milnet, criterion)
    
        score = get_current_score(avg_score, aucs)
        if score > fold_best_score and round_com > 3:
            fold_best_score = score
            best_ac = avg_score
            best_auc = aucs
            save_model(args, fold, 0, save_path, milnet, thresholds_optimal)

if __name__ == '__main__':
    main()
