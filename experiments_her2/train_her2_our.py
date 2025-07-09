import os
import sys
import warnings
import argparse
import time
import glob
import copy
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from collections import OrderedDict

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add custom libraries
sys.path.append('../')
from config import *
from Federated.FedAvg import FedWeightAvg, FedWeightAvgBatchNorm, FedProx
from Federated.LocalUpdate import *
from utils.init_model import *
from utils.validation import *


def main():
    parser = argparse.ArgumentParser(description='Train F-MIL')

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--stop_epochs', default=10, type=int)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(5,))
    parser.add_argument('--weight_decay', default=5e-3, type=float)
    parser.add_argument('--dataset', default='her2', type=str)
    parser.add_argument('--model', default='HistoFL', type=str)
    parser.add_argument('--dropout_patch', default=0.0, type=float)
    parser.add_argument('--dropout_node', default=0.0, type=float)
    parser.add_argument('--non_linearity', default=1.0, type=float)
    parser.add_argument('--average', type=bool, default=False)
    parser.add_argument('--eval_scheme', default='5-fold-cv', type=str)

    parser.add_argument('--backbone', default='dino', type=str, choices=['resnet', 'dino'])
    parser.add_argument('--style', default='Our')
    parser.add_argument('--federated', default='FedAvg', type=str, choices=['FedAvg', 'FedBN', 'FedProx'])
    parser.add_argument('--round_fl', default=20, type=int)
    parser.add_argument('--train_percentage', default=100, type=int)

    args = parser.parse_args()

    args.feats_size = {'resnet': 1024, 'dino': 384}[args.backbone]
    args.num_classes = {'her2': 2, 'tcga_rcc': 3}[args.dataset]
    args.batch_norm = {'FedAvg': False, 'FedBN': True, 'FedProx': False}[args.federated]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu_index)
    args.device = torch.device(f'cuda:{args.gpu_index[0]}')

    if args.model == 'HistoFL':
        import models.histofl as mil
    elif args.model == 'frmil':
        import models.frmil as mil

    print('FEDERATED LEARNING')

    df_seen_train = pd.read_csv(os.path.join('..', 'labels', args.dataset, f'train_seen_{args.train_percentage}.csv'))
    df_seen_test = pd.read_csv(os.path.join('..', 'labels', args.dataset, 'test_seen.csv'))

    df_seen_train_grouped = df_seen_train.groupby('center')
    df_seen_test_grouped = df_seen_test.groupby('center')
    distinct_centers_seen = df_seen_train['center'].unique()

    if args.federated in ['FedBN', 'FedProx']:
        args.experiment = f'{args.backbone}_{args.federated}_{args.train_percentage}'
    else:
        args.experiment = f'{args.backbone}_{args.style}_{args.train_percentage}'

    print(f'Dataset: {args.dataset}, Experiment: {args.experiment}')
    save_path = os.path.join('..', 'weights_uni', args.model, args.dataset, args.experiment)
    os.makedirs(save_path, exist_ok=True)

    milnet, criterion, optimizer, scheduler = init_model(mil, args)
    milnet.train()

    clients = {}
    cached_statistics = {}

    for train_groups, test_groups in zip(df_seen_train_grouped, df_seen_test_grouped):
        train_center, train_group = train_groups
        test_center, test_group = test_groups

        client = LocalUpdate(
            args,
            bag_train=train_group,
            bag_test=test_group,
            center_id=train_center,
            local_milnet=copy.deepcopy(milnet).cuda()
        )
        clients[train_center] = client
        cached_statistics[train_center] = client.extract_statistics(args, train_group)

        print(f'Center: {train_center}, Train: {train_group.shape}, Test: {test_group.shape}')

    # Combine statistics excluding own center
    other_center_stats = {}
    for center in cached_statistics:
        mu_all, var_all = [], []
        for other_center in cached_statistics:
            if other_center != center:
                mu_all.extend(cached_statistics[other_center]['bag_mu'])
                var_all.extend(cached_statistics[other_center]['bag_var'])
        other_center_stats[center] = {'bag_mu': mu_all, 'bag_var': var_all}

    round_durations = []

    for fold in range(5):
        fold_best_score = 0
        best_ac = 0
        best_auc = 0
        milnet, criterion, optimizer, scheduler = init_model(mil, args)
        milnet.train()

        for round_com in range(args.round_fl):
            start_time = time.time()
            w_locals, train_loss_locals = [], []
            total_client_size = []

            for center in distinct_centers_seen:
                w_local, loss_local, len_local_paths = clients[center].train(
                    args,
                    milnet=copy.deepcopy(milnet).cuda(),
                    cached_statistics=other_center_stats[center]
                )
                w_locals.append(w_local)
                train_loss_locals.append(loss_local)
                total_client_size.append(len_local_paths)

            # Aggregate global model
            w_glob = FedWeightAvg(milnet, w_locals)
            milnet.load_state_dict(w_glob)

            print('Evaluate All Institutes:')
            val_loss, avg_score, aucs, thresholds_optimal = test(args, df_seen_test, milnet, criterion)

            score = get_current_score(avg_score, aucs)
            if score > fold_best_score and round_com > 3:
                fold_best_score = score
                best_ac = avg_score
                best_auc = aucs
                save_model(args, fold, 0, save_path, milnet, thresholds_optimal)
                milnet.train()

            round_durations.append(time.time() - start_time)

    avg_duration = sum(round_durations) / len(round_durations)
    print(f'Average training time per round: {avg_duration:.2f} seconds')


if __name__ == '__main__':
    main()
