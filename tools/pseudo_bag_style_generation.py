import os
import argparse
import numpy as np
import pandas as pd
import torch

def wasserstein_distance_2d(mean1, std1, mean2, std2):
    """Compute the 2-Wasserstein distance between two 1D Gaussians."""
    return np.sqrt((mean1 - mean2)**2 + (std1 - std2)**2)

def wasserstein_kmeans(features, num_clusters, max_iters=1000, tol=1e-4):
    """Cluster [N, 2] features (mean, std) using 2-Wasserstein distance."""
    means = features[:, 0]
    stds = features[:, 1]

    centroid_means = np.random.choice(means, num_clusters, replace=False)
    centroid_stds = np.random.choice(stds, num_clusters, replace=False)

    for iteration in range(max_iters):
        distances = np.array([
            [wasserstein_distance_2d(m, s, cm, cs) for cm, cs in zip(centroid_means, centroid_stds)]
            for m, s in zip(means, stds)
        ])
        cluster_ids = np.argmin(distances, axis=1)

        new_means = np.array([
            means[cluster_ids == k].mean() if np.any(cluster_ids == k) else centroid_means[k]
            for k in range(num_clusters)
        ])
        new_stds = np.array([
            stds[cluster_ids == k].mean() if np.any(cluster_ids == k) else centroid_stds[k]
            for k in range(num_clusters)
        ])

        if np.linalg.norm(centroid_means - new_means) < tol and np.linalg.norm(centroid_stds - new_stds) < tol:
            print(f"Converged at iteration {iteration}")
            break

        centroid_means, centroid_stds = new_means, new_stds

    return cluster_ids, centroid_means, centroid_stds

def main(args):
    csv_path = os.path.join('..', 'labels', args.dataset, 'train.csv')
    df = pd.read_csv(csv_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, row in df.iterrows():
        bag_path = row['patient']
        if not os.path.isfile(bag_path):
            print(f"[Skip] File not found: {bag_path}")
            continue

        style_path = bag_path.replace('/pt/', '/style_pt/')
        os.makedirs(os.path.dirname(style_path), exist_ok=True)

        features = torch.load(bag_path, map_location=device)
        feature_dim = 1024 if args.FEATS_TYPE == 'resnet' else 384
        features = features[:, :feature_dim]  # [N, D]

        patch_means = features.mean(dim=1)  # [N]
        patch_stds = features.std(dim=1)    # [N]
        style_features = torch.stack([patch_means, patch_stds], dim=1).cpu().numpy()  # [N, 2]

        cluster_ids, centroid_means, centroid_stds = wasserstein_kmeans(
            style_features, args.num_pseudo_style)

        torch.save({
            'pseudo_styles': np.stack([centroid_means, centroid_stds], axis=1).astype(np.float32),  # [K, 2]
            'cluster_ids': cluster_ids.astype(np.int32),  # [N]
        }, style_path)

        print(f"[{idx+1}/{len(df)}] Saved pseudo styles to: {style_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo Bag Style Generation (HistoFS, mean/std notation).")
    parser.add_argument("--FEATS_TYPE", default='ssl_vit', choices=['ssl_vit', 'resnet'], help="Feature type")
    parser.add_argument("--dataset", default='c17', choices=['c17', 'tcga_rcc', 'her2'], help="Dataset name")
    parser.add_argument("--num_pseudo_style", default=5, type=int, help="Number of pseudo styles (K)")
    args = parser.parse_args()

    main(args)
