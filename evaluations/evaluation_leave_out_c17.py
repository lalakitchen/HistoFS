

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

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

''' Own library '''
own_dir = os.path.join('../')
sys.path.append(own_dir)

from config import *
from Federated.FedAvg import FedWeightAvg, FedWeightAvgBatchNorm
from Federated.LocalUpdate import *
from utils.init_model import *
from utils.validation import *

def main():
    parser = argparse.ArgumentParser(description='Train F-MIL')
    
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=10, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(5,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--dataset', default='c17', type=str, help='Dataset folder name')
    parser.add_argument('--model', default='HistoFL', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')

    parser.add_argument("--backbone",default='dino', type=str, choices=['resnet', 'dino'])
    parser.add_argument("--style",default='Our', type=str,)
    parser.add_argument("--federated",default='FedAvg', type=str, choices=['FedAvg', 'FedBN', 'FedProx'])
    parser.add_argument("--round_fl",default=20, type=int)
    parser.add_argument("--train_percentage",default=100, type=int)
    parser.add_argument('--Auth', default=False, type=bool)
    parser.add_argument("--leave_out",default=3, type=int)


    args = parser.parse_args()
    args.feats_size = {'resnet': 1024, 'dino': 384}[args.backbone]
    args.num_classes = {'her2': 2, 'tcga_rcc': 3, 'c17':2}[args.dataset]
    args.batch_norm = {'FedAvg': False, 'FedBN': True, 'FedProx': False}[args.federated]
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    args.device = torch.device('cuda:{}'.format(6))

    gpu_ids = tuple(args.gpu_index)



   

    if args.model =='HistoFL':
        import models.histofl as mil
    elif args.model =='PointTransformerFL':
        import models.point_transformer as mil
    elif args.model =='dsmil':
        import models.dsmil as mil
    elif args.model =='frmil':
        import models.frmil as mil


    print(' FEDERATED LEARNING ')
    
    df_seen_train =  pd.read_csv(os.path.join('..', 'labels', args.dataset, f'train_seen_leave_{args.leave_out}.csv'))
    df_seen_test =  pd.read_csv(os.path.join('..', 'labels', args.dataset, f'test_seen_leave_{args.leave_out}.csv'))
        
    df_seen_train_grouped = df_seen_train.groupby('center')
    df_seen_test_grouped = df_seen_test.groupby('center')
    
    distinct_centers_seen = df_seen_train['center'].unique()
    print(distinct_centers_seen)

  
   # Set experiment name
    if args.Auth:
        args.experiment = f"{args.backbone}_w_auth_leave_out_{args.leave_out}"
    else:
        args.experiment = f"{args.backbone}_wo_auth_leave_out_{args.leave_out}"

    print(f'Dataset: {args.dataset}, Experiment: {args.experiment}')

    # Load model checkpoints
    weights = sorted(glob.glob(os.path.join('..', 'model_checkpoints', args.model, args.dataset, args.experiment, '*.pth')))
    



    print(f'Dataset : {args.dataset}, experiment : {args.experiment}')
   
    milnet, criterion, optimizer, scheduler = init_model(mil, args)
    milnet.train()
    w_glob = milnet.state_dict()

    accuracy_total, auc_total = [], []
    for weight in weights:
        state_dict_weights = torch.load(weight)
        milnet.load_state_dict(state_dict_weights, strict=True)
        milnet.eval()
        
        for center, df_seen_test_center in df_seen_test_grouped:
            print(f'\r center_{center}')
            bAcc, f1, auc = evaluation(args, df_seen_test_center, milnet, criterion)
            auc_avg = sum(auc) / len(auc) if auc else 0.0
            print(f'    Balanced Acc: {bAcc*100:.2f}%, F1: {f1*100:.2f}%, AUC: {auc_avg:.4f}')
           
        


if __name__ == '__main__':
    main()

