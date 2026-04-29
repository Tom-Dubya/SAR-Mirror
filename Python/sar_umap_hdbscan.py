import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path

import umap
import hdbscan
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors

from Models.GetModel import get_model


def extract_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features = model.forward_features(images)

            all_features.append(features.cpu())
            all_labels.append(labels)
            all_images.append(images.cpu())

    X = torch.cat(all_features, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    imgs = torch.cat(all_images, dim=0)

    return X, y, imgs


def show_neighbors(query_idx, indices, imgs, labels, paths, save_dir):
    neighbors = indices[query_idx]

    # Get query image path info
    query_path = Path(paths[query_idx])
    class_name = query_path.parent.name
    base_name = query_path.stem

    # Create output directory per class
    out_dir = Path(save_dir) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 3))

    for i, idx in enumerate(neighbors):
        plt.subplot(1, len(neighbors), i + 1)

        img = imgs[idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        plt.imshow(img)
        plt.title(f"L:{labels[idx]}")
        plt.axis("off")

    plt.suptitle(f"Query: {base_name}")

    # Save figure
    save_path = out_dir / f"{base_name}_neighbors.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def print_cluster_purity(cluster_labels, y):
    print("\n--- Cluster Purity Analysis ---")

    cluster_to_labels = defaultdict(list)

    for i, c in enumerate(cluster_labels):
        if c == -1:
            continue
        cluster_to_labels[c].append(y[i])

    for c, labels in cluster_to_labels.items():
        counts = Counter(labels)
        majority = max(counts.values())
        purity = majority / len(labels)

        print(
            f"Cluster {c}: "
            f"size={len(labels)}, "
            f"purity={purity:.3f}, "
            f"distribution={dict(counts)}"
        )


def main():

    model_name = "Sample"

    # model_path = "Output/raw_outputs/python_cnn_output_standard_70-30_1.pt"
    # test_dir = "../../../datasets/70-30/equiv_subst_standard_70-30/test"
    # save_dir = "neighbor_outputs/70-30/standard"
    #
    # model_path = "Output/raw_outputs/python_cnn_output_qpm_70-30_1.pt"
    # test_dir = "../../../datasets/70-30/equiv_subst_qpm_70-30/test"
    # save_dir = "neighbor_outputs/70-30/qpm"
    #
    # model_path = "Output/raw_outputs/python_cnn_output_decibel_70-30_1.pt"
    # test_dir = "../../../datasets/70-30/equiv_subst_decibel_70-30/test"
    # save_dir = "neighbor_outputs/70-30/decibel"
    #
    model_path = "Output/raw_outputs/python_cnn_output_correlated_70-30_1.pt"
    test_dir = "../../../datasets/70-30/equiv_subst_correlated_70-30/test"
    save_dir = "neighbor_outputs/70-30/correlated"

    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir_path = Path(test_dir)
    num_classes = sum(1 for d in test_dir_path.iterdir() if d.is_dir())

    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=model.test_transform
    )

    # 🔥 Get file paths
    paths = [p for p, _ in test_dataset.samples]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("Extracting features...")
    X, y, imgs = extract_features(model, test_loader, device)
    print("Feature shape:", X.shape)

    # UMAP (CLUSTERING SPACE)
    USE_RAW = False

    if USE_RAW:
        X_cluster = X
        print("Clustering on RAW CNN features...")
    else:
        print("Running UMAP for clustering (10D)...")
        umap_cluster = umap.UMAP(
            n_components=10,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )
        X_cluster = umap_cluster.fit_transform(X)

    # HDBSCAN clustering
    print("Running HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(20, len(X) // 200),
        min_samples=10,
        cluster_selection_method="eom"
    )

    cluster_labels = clusterer.fit_predict(X_cluster)

    # Cluster summaries
    print("\n--- Cluster summaries ---")

    for k in set(cluster_labels):
        if k == -1:
            continue
        idxs = np.where(cluster_labels == k)[0]
        label_counts = Counter(y[idxs])

        print(f"Cluster {k}: size={len(idxs)}, labels={dict(label_counts)}")

    # PURITY ANALYSIS
    print_cluster_purity(cluster_labels, y)

    # Nearest Neighbors
    print("\nBuilding Nearest Neighbor index...")

    nn = NearestNeighbors(n_neighbors=6)
    nn.fit(X_cluster)

    distances, indices = nn.kneighbors(X_cluster)

    print("\nSaving nearest neighbors...")

    samples_per_class = 3
    sample_indices = []

    class_to_indices = defaultdict(list)

    for i, label in enumerate(y):
        class_to_indices[label].append(i)

    for cls, idxs in class_to_indices.items():
        chosen = np.random.choice(idxs, min(samples_per_class, len(idxs)), replace=False)
        sample_indices.extend(chosen)

    for query_idx in sample_indices:
        print(f"\nQuery index: {query_idx}, label: {y[query_idx]}")
        print("Neighbor labels:", y[indices[query_idx]])

        show_neighbors(query_idx, indices, imgs, y, paths, save_dir)

    # UMAP VISUALIZATION (2D ONLY)
    print("Running UMAP for visualization (2D)...")

    umap_vis = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    X_vis = umap_vis.fit_transform(X)

    vis_dir = Path("clustering_outputs/70-30")
    vis_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X_vis[:, 0],
        X_vis[:, 1],
        c=cluster_labels,
        cmap="Spectral",
        s=10
    )

    plt.title("Embedding Clusters (HDBSCAN on 10D UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.colorbar(scatter, label="Cluster ID")

    save_path = vis_dir / "correlated_umap_hdbscan_clusters.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    print(f"Saved clustering visualization to: {save_path}")

    # FINAL STATS
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise = np.sum(cluster_labels == -1)

    print("\n--- Results ---")
    print("Clusters found:", num_clusters)
    print("Noise points:", noise)
    print("Total samples:", len(cluster_labels))


if __name__ == "__main__":
    main()