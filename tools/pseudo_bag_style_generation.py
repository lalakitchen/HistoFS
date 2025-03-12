import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ot  # POT library for optimal transport and Wasserstein distance

def wasserstein_distance_2d(mean1, std1, mean2, std2):
    """Compute the 2-Wasserstein distance between two normal distributions."""
    return np.sqrt((mean1 - mean2) ** 2 + (std1 - std2) ** 2)

def wasserstein_kmeans(features, n_clusters, max_iters=1000, tol=1e-4):
    means, stds = features[:, 0], features[:, 1]
    centroids_means, centroids_stds = np.random.choice(means, n_clusters, replace=False), np.random.choice(stds, n_clusters, replace=False)
    
    for iteration in range(max_iters):
        distances = np.array([[wasserstein_distance_2d(m, s, cm, cs) for cm, cs in zip(centroids_means, centroids_stds)] for m, s in zip(means, stds)])
        cluster_assignments = np.argmin(distances, axis=1)
        
        new_centroids_means = np.array([means[cluster_assignments == k].mean() for k in range(n_clusters)])
        new_centroids_stds = np.array([stds[cluster_assignments == k].mean() for k in range(n_clusters)])
        
        if np.linalg.norm(centroids_means - new_centroids_means) < tol and np.linalg.norm(centroids_stds - new_centroids_stds) < tol:
            print("Convergence reached.")
            break
        
        centroids_means, centroids_stds = new_centroids_means, new_centroids_stds
    
    return cluster_assignments, centroids_means, centroids_stds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo Bag Style Generation.")
    parser.add_argument("--FEATS_TYPE", default='ssl_vit', type=str, choices=['ssl_vit', 'resnet'], help="Feature type")
    parser.add_argument("--dataset", default='c17', type=str, choices=['c17', 'tcga_rcc', 'her2'], help="Dataset selection")
    parser.add_argument("--NUM_PSEUDO_STYLE", default=5, type=int, help="Number of pseudo styles")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join('..', 'labels', args.dataset, 'train.csv'))
    tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    for _, item in df.iterrows():
        bag_item = item['patient']
        if os.path.isfile(bag_item):
            style_path = bag_item.replace('/pt/', '/style_pt/')
            os.makedirs(os.path.dirname(style_path), exist_ok=True)
            stacked_data = torch.load(bag_item, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
            feature_dim = 1024 if args.FEATS_TYPE == 'resnet' else 384
            bag_features = tensor_type(stacked_data[:, :feature_dim])
            
            features_np = torch.stack((bag_features.mean(dim=1), bag_features.std(dim=1)), dim=1).cpu().numpy()
            cluster_assignments, centroids_means, centroids_stds = wasserstein_kmeans(features_np, args.NUM_PSEUDO_STYLE)
            
            torch.save({'centroids_means': centroids_means, 'centroids_stds': centroids_stds}, style_path)
