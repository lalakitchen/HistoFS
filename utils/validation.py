import os
import sys
import json
import math
import copy
import random
import warnings
import argparse
import itertools
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mode
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score,
    hamming_loss, precision_score, recall_score, average_precision_score,
    f1_score, cohen_kappa_score, precision_recall_curve
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=UserWarning, message="The torch.cuda.*DtypeTensor constructors are no longer recommended")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Own library
own_dir = os.path.join('..')
sys.path.append(own_dir)
from config import *

def get_bag_feats(csv_file_df):
    try:
        feats = shuffle(pd.read_csv(csv_file_df, header=None)).reset_index(drop=True).to_numpy()
    except Exception:
        feats = np.array([])
    return feats


def dropout_patches(feats, p):
    n = feats.size(0)
    selected_indices = torch.randperm(n)[:int(n * p)]
    return feats[selected_indices]


def inverse_convert_label(labels):
    # one-hot decoding
    return labels if labels.ndim == 1 else np.argmax(labels, axis=1)

    
def evaluation(args, test_df, milnet, criterion, thresholds=None):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    tensor_type = torch.cuda.FloatTensor
    bags_seen_test = test_df.values.tolist()

    length_data = 0
    bag_statistics = {'bag_mu': [], 'bag_var': []}

    with torch.no_grad():
        for i, item in enumerate(bags_seen_test):
            file_name = os.path.basename(item[0]).replace('.csv', '.pt')
        
            if 'c17' in args.dataset:
                bag_item = os.path.join(C17_PATH, 'pt', args.backbone, file_name)
        
            elif 'tcga_rcc' in args.dataset:
                # Adjust for .svs.pt if needed
                svs_file = os.path.basename(item[0]) + '.svs.pt'
                bag_item = os.path.join(TCGA_RCC_PATH, 'pt', args.backbone, svs_file)
        
            elif 'her2' in args.dataset:
                if 'herohe' in item[0]:
                    base = TCGA_HEROHE_PATH
                elif 'brca' in item[0]:
                    base = TCGA_BRCA_PATH
                elif 'yale' in item[0]:
                    base = TCGA_YALE_PATH
                else:
                    continue  # unknown subtype
                bag_item = os.path.join(base, 'pt', args.backbone, file_name)
        
            else:
                continue  # unsupported dataset
        
            if not os.path.isfile(bag_item):
                continue

            stacked_data = torch.load(bag_item, map_location='cuda:0')
            bag_label = item[1]

            if args.num_classes != 1:
                label = np.zeros(args.num_classes)
                if int(bag_label) <= (len(label) - 1):
                    label[int(bag_label)] = 1
                bag_label = Variable(torch.FloatTensor([label]).cuda())

            bag_features = tensor_type(stacked_data[:, :args.feats_size])
            bag_features = dropout_patches(bag_features, 1 - args.dropout_patch)
            bag_features = bag_features.view(-1, args.feats_size)

            
            if args.model == 'HistoFL':
                bag_prediction, _, _, _ = milnet(bag_features)
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model == 'frmil':
                bag_prediction, _, _ = milnet(bag_features)
                loss = F.cross_entropy(bag_prediction, bag_label)
       
            else:
                continue


            total_loss += loss.item()
            sys.stdout.write(f'\r Testing bag [{i + 1}/{len(test_df)}] bag loss: {loss.item():.4f}')

            pred_sigmoid = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            if args.model == 'dsmil' and args.average:
                pred_sigmoid = (pred_sigmoid + torch.sigmoid(max_prediction).squeeze().cpu().numpy()) / 2

            test_predictions.append(pred_sigmoid)
            test_labels.append(bag_label.squeeze().cpu().numpy().astype(int))
            length_data += 1

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes)
    if thresholds:
        thresholds_optimal = thresholds

    if args.num_classes == 1:
        test_predictions = (test_predictions >= thresholds_optimal[0]).astype(int)
        test_labels = test_labels.squeeze()
    else:
        for i in range(args.num_classes):
            test_predictions[:, i] = (test_predictions[:, i] >= thresholds_optimal[i]).astype(int)

    avg_score = np.mean([np.array_equal(test_labels[i], test_predictions[i]) for i in range(length_data)])
    y_pred = inverse_convert_label(test_predictions)
    y_true = inverse_convert_label(test_labels)
    bAcc = balanced_accuracy_score(y_true, y_pred)
    mAP = np.mean([average_precision_score(test_labels[:, i], test_predictions[:, i]) for i in range(args.num_classes)])
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f'bAcc {bAcc:.4f}, mAP {mAP:.4f}, f1 {f1:.4f}')

    return bAcc, f1, auc_value


def test(args, test_df, milnet, criterion, thresholds=None, return_predictions=False):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    tensor_type = torch.cuda.FloatTensor
    bags_seen_test = test_df.values.tolist()

    length_data = 0
    bag_statistics = {'bag_mu': [], 'bag_var': []}

    with torch.no_grad():
        for i, item in enumerate(bags_seen_test):
            file_name = os.path.basename(item[0]).replace('.csv', '.pt')
        
            if 'c17' in args.dataset:
                bag_item = os.path.join(C17_PATH, 'pt', args.backbone, file_name)
        
            elif 'tcga_rcc' in args.dataset:
                # Adjust for .svs.pt if needed
                svs_file = os.path.basename(item[0]) + '.svs.pt'
                bag_item = os.path.join(TCGA_RCC_PATH, 'pt', args.backbone, svs_file)
        
            elif 'her2' in args.dataset:
                if 'herohe' in item[0]:
                    base = TCGA_HEROHE_PATH
                elif 'brca' in item[0]:
                    base = TCGA_BRCA_PATH
                elif 'yale' in item[0]:
                    base = TCGA_YALE_PATH
                else:
                    continue  # unknown subtype
                bag_item = os.path.join(base, 'pt', args.backbone, file_name)
        
            else:
                continue  # unsupported dataset
        
            if not os.path.isfile(bag_item):
                continue

            stacked_data = torch.load(bag_item, map_location='cuda:0')
            bag_label = item[1]

            if args.num_classes != 1:
                label = np.zeros(args.num_classes)
                if int(bag_label) <= (len(label) - 1):
                    label[int(bag_label)] = 1
                bag_label = Variable(torch.FloatTensor([label]).cuda())

            bag_features = tensor_type(stacked_data[:, :args.feats_size])
            bag_features = dropout_patches(bag_features, 1 - args.dropout_patch)
            bag_features = bag_features.view(-1, args.feats_size)

            
            if args.model == 'HistoFL':
                bag_prediction, _, _, _ = milnet(bag_features)
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model == 'frmil':
                bag_prediction, _, _ = milnet(bag_features)
                loss = F.cross_entropy(bag_prediction, bag_label)
       
            else:
                continue


            total_loss += loss.item()
            sys.stdout.write(f'\r Testing bag [{i + 1}/{len(test_df)}] bag loss: {loss.item():.4f}')

            pred_sigmoid = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            if args.model == 'dsmil' and args.average:
                pred_sigmoid = (pred_sigmoid + torch.sigmoid(max_prediction).squeeze().cpu().numpy()) / 2

            test_predictions.append(pred_sigmoid)
            test_labels.append(bag_label.squeeze().cpu().numpy().astype(int))
            length_data += 1

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes)
    if thresholds:
        thresholds_optimal = thresholds

    if args.num_classes == 1:
        test_predictions = (test_predictions >= thresholds_optimal[0]).astype(int)
        test_labels = test_labels.squeeze()
    else:
        for i in range(args.num_classes):
            test_predictions[:, i] = (test_predictions[:, i] >= thresholds_optimal[i]).astype(int)

    avg_score = np.mean([np.array_equal(test_labels[i], test_predictions[i]) for i in range(length_data)])
    y_pred = inverse_convert_label(test_predictions)
    y_true = inverse_convert_label(test_labels)
    bAcc = balanced_accuracy_score(y_true, y_pred)
    mAP = np.mean([average_precision_score(test_labels[:, i], test_predictions[:, i]) for i in range(args.num_classes)])
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f'bAcc {bAcc:.4f}, mAP {mAP:.4f}, f1 {f1:.4f}')

    return total_loss / length_data, bAcc, auc_value, thresholds_optimal

    
def multi_label_roc(labels, predictions, num_classes):
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)

    aucs, thresholds, thresholds_optimal = [], [], []

    for c in range(num_classes):
        label = labels[:, c]
        pred = predictions[:, c]

        try:
            fpr, tpr, thresh = roc_curve(label, pred)
            _, _, thresh_opt = optimal_thresh(fpr, tpr, thresh)
            auc = roc_auc_score(label, pred)
        except ValueError as e:
            if "Only one class present" in str(e):
                print(f"[Warning] ROC AUC undefined for class {c}. Setting AUC to 1.")
                auc, thresh_opt = 1.0, 0.5
            else:
                raise

        aucs.append(auc)
        thresholds.append(thresh)
        thresholds_optimal.append(thresh_opt)

    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1e-8)
    idx = np.argmin(loss)
    return fpr[idx], tpr[idx], thresholds[idx]


def print_epoch_info(epoch, center_id, args, train_loss, val_loss, avg_score, aucs):
    auc_str = ' | '.join(f'class-{i}>>{auc:.3f}' for i, auc in enumerate(aucs))
    print(f'\n[Center {center_id}] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} '
          f'| Avg Score: {avg_score:.4f} | AUC: {auc_str}')


def get_current_score(avg_score, aucs):
    return (np.mean(aucs) + avg_score) / 2


