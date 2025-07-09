import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
import warnings
import argparse
import glob
import copy

import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# Add local module directory
sys.path.append('../')

# Local libraries
from config import *
from Federated.FedAvg import FedWeightAvg, FedWeightAvgBatchNorm
from Federated.LocalUpdate import *
from utils.init_model import *
from utils.validation import *


def main():
    parser = argparse.ArgumentParser(description='Federated MIL Evaluation')

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--stop_epochs', default=10, type=int)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,))
    parser.add_argument('--weight_decay', default=5e-3, type=float)
    parser.add_argument('--model', default='HistoFL', type=str)
    parser.add_argument('--dropout_patch', default=0.0, type=float)
    parser.add_argument('--dropout_node', default=0.0, type=float)
    parser.add_argument('--non_linearity', default=1.0, type=float)
    parser.add_argument('--average', type=bool, default=False)

    parser.add_argument('--backbone', default='dino', type=str)
    parser.add_argument('--dataset', default='her2', type=str)
    parser.add_argument('--experiment', default='', type=str)
    parser.add_argument('--round_fl', default=10, type=int)
    parser.add_argument('--style', default='Our', type=str)
    parser.add_argument('--Auth', default=True, type=bool)
    parser.add_argument('--federated', default='FedAvg', type=str, choices=['FedAvg', 'FedBN'])

    args = parser.parse_args()

    args.feats_size = {'resnet': 1024, 'dino': 384}[args.backbone]
    args.num_classes = {'her2': 2, 'tcga_rcc': 3}[args.dataset]
    args.batch_norm = {'FedAvg': False, 'FedBN': True}[args.federated]
    args.device = torch.device(f'cuda:{args.gpu_index[0]}' if torch.cuda.is_available() else 'cpu')

    # Load model
    if args.model == 'HistoFL':
        import models.histofl as mil
    elif args.model == 'frmil':
        import models.frmil as mil

    print('FEDERATED LEARNING')

    # Load test data
    df_seen_test = pd.read_csv(os.path.join('..', 'labels', args.dataset, 'test_seen.csv'))
    df_seen_test_grouped = df_seen_test.groupby('center')

    # Set experiment name
    if args.Auth:
        args.experiment = f"{args.backbone}_w_auth"
    else:
        args.experiment = f"{args.backbone}_wo_auth"

    print(f'Dataset: {args.dataset}, Experiment: {args.experiment}')

    # Load model checkpoints
    weights = sorted(glob.glob(os.path.join('..', 'model_checkpoints', args.model, args.dataset, args.experiment, '*.pth')))
    if not weights:
        print("No model weights found.")
        return

    print(f'Found {len(weights)} weight files.')

    # Initialize model
    milnet, criterion, optimizer, scheduler = init_model(mil, args)
    milnet.train()

    # Evaluation
    for weight_path in weights:
        print(f'\nEvaluating: {os.path.basename(weight_path)}')
        state_dict = torch.load(weight_path, map_location=args.device)
        milnet.load_state_dict(state_dict, strict=True)
        milnet.eval()

        for center, df_center in df_seen_test_grouped:
            print(f'  Center {center}')
            bAcc, f1, auc = evaluation(args, df_center, milnet, criterion)
            auc_avg = sum(auc) / len(auc) if auc else 0.0
            print(f'    Balanced Acc: {bAcc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc_avg:.4f}')


if __name__ == '__main__':
    main()
