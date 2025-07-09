import os
import sys
import argparse
import warnings
import torch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Add local libraries
sys.path.append('../')
from config import *


def wasserstein_distance_2d(mean1, var1, mean2, var2):
    """Compute the 2-Wasserstein distance between two 1D normal distributions using variance."""
    return np.sqrt((mean1 - mean2) ** 2 + (np.sqrt(var1) - np.sqrt(var2)) ** 2)


def wasserstein_kmeans(features, n_clusters, max_iters=100, tol=1e-4):
    means, vars_ = features[:, 0], features[:, 1]
    centroids_means = np.random.choice(means, n_clusters, replace=False)
    centroids_vars = np.random.choice(vars_, n_clusters, replace=False)

    for iteration in range(max_iters):
        distances = np.zeros((features.shape[0], n_clusters))
        for i, (m, v) in enumerate(zip(means, vars_)):
            for j, (cm, cv) in enumerate(zip(centroids_means, centroids_vars)):
                distances[i, j] = wasserstein_distance_2d(m, v, cm, cv)

        assignments = np.argmin(distances, axis=1)

        new_means = np.array([means[assignments == k].mean() for k in range(n_clusters)])
        new_vars = np.array([vars_[assignments == k].mean() for k in range(n_clusters)])

        mean_change = np.linalg.norm(centroids_means - new_means)
        var_change = np.linalg.norm(centroids_vars - new_vars)

        sys.stdout.write(f"Iteration {iteration + 1}/{max_iters}: ")
        sys.stdout.write(f"Mean Δ: {mean_change:.4f}, Var Δ: {var_change:.4f}\n")

        if mean_change < tol and var_change < tol:
            sys.stdout.write("Convergence reached.\n")
            break

        centroids_means, centroids_vars = new_means, new_vars

    return assignments, centroids_means, centroids_vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Wasserstein KMeans using variance.")
    parser.add_argument("--FEATS_TYPE", default='uni', type=str, choices=['dino', 'resnet', 'uni'])
    parser.add_argument("--dataset", default='tcga_rcc', type=str, choices=['c17', 'tcga_rcc', 'her2'])
    parser.add_argument("--NUM_CLUSTER", default=5, type=int)
    parser.add_argument("--leave_out", type=int)  # included for compatibility
    args = parser.parse_args()

    if args.dataset == 'tcga_rcc':
        DATASET_PATH = TCGA_RCC_PATH
    elif args.dataset == 'c17':
        DATASET_PATH = C17_PATH
    else:
        raise ValueError("Unsupported dataset.")

    tensor_type = torch.cuda.FloatTensor
    label_path = os.path.join('..', 'labels', args.dataset, 'train_seen.csv')
    df = pd.read_csv(label_path)

    for _, item in df.iterrows():
        original_path = item['patient']
        prefix = os.path.basename(os.path.dirname(original_path))
        bag_item = original_path.replace(prefix, f"pt/{args.FEATS_TYPE}").replace('svs.csv', 'pt')

        if not os.path.isfile(bag_item):
            continue

        style_path = bag_item.replace('/pt/', '/style_pt/')
        os.makedirs(os.path.dirname(style_path), exist_ok=True)

        data = torch.load(bag_item, map_location='cuda:0')
        if args.FEATS_TYPE == 'resnet':
            features = data[:, :1024]
        elif args.FEATS_TYPE == 'dino':
            features = data[:, :384]
        else:
            features = data

        features = tensor_type(features)
        instance_mean = features.mean(dim=1)
        instance_var = features.var(dim=1, unbiased=False)
        stats = torch.stack((instance_mean, instance_var), dim=1).cpu().numpy()

        assignments, centroids_means, centroids_vars = wasserstein_kmeans(stats, args.NUM_CLUSTER)

        torch.save({
            'centroids_means': centroids_means,
            'centroids_vars': centroids_vars
        }, style_path)
