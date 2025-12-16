"""Visualization and analysis functions for metric learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def compute_embeddings_and_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute embeddings and labels for all samples.

    Args:
        model: embedding model.
        data_loader: DataLoader with (images, labels) batches.
        device: device to run on.

    Returns:
        (embeddings, labels) as tensors.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing embeddings", leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
            else:
                raise ValueError("Expected batch to be (inputs, labels) tuple or list")

            inputs = inputs.to(device)
            labels = labels.to(device)

            embeddings = model(inputs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


def compute_confusion_matrix(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute confusion matrix by finding nearest neighbor class for each sample.

    Args:
        embeddings: shape (N, dim)
        labels: shape (N,)
        metric: distance metric ("euclidean" or "cosine").

    Returns:
        confusion matrix of shape (num_classes, num_classes).
    """
    if metric == "euclidean":
        # Compute Euclidean distances: (N, N)
        distances = torch.cdist(embeddings, embeddings, p=2)
    elif metric == "cosine":
        # Normalize embeddings
        norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        # Cosine distance = 1 - cosine similarity
        distances = 1 - torch.mm(norm, norm.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # For each sample, find the nearest neighbor (excluding itself)
    # Set diagonal to infinity to exclude self
    distances[torch.eye(distances.shape[0], dtype=torch.bool)] = float("inf")

    # Get index of nearest neighbor for each sample
    nearest_indices = torch.argmin(distances, dim=1)
    predicted_labels = labels[nearest_indices]

    # Compute confusion matrix
    num_classes = len(torch.unique(labels))
    cm = confusion_matrix(
        labels.numpy(), predicted_labels.numpy(), labels=list(range(num_classes))
    )

    return cm


def compute_classification_report(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "euclidean",
    target_names: list[str] | None = None,
) -> str:
    """Compute classification report based on nearest-neighbor prediction.

    Args:
        embeddings: shape (N, dim)
        labels: shape (N,)
        metric: distance metric ("euclidean" or "cosine").
        target_names: optional list of class names for pretty printing.

    Returns:
        classification report as a formatted string.
    """
    if metric == "euclidean":
        distances = torch.cdist(embeddings, embeddings, p=2)
    elif metric == "cosine":
        norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        distances = 1 - torch.mm(norm, norm.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Exclude self by setting diagonal to infinity
    distances[torch.eye(distances.shape[0], dtype=torch.bool)] = float("inf")

    # Get nearest neighbor predictions
    nearest_indices = torch.argmin(distances, dim=1)
    predicted_labels = labels[nearest_indices]

    # Compute classification report
    report = classification_report(
        labels.numpy(),
        predicted_labels.numpy(),
        target_names=target_names,
        zero_division=0,
    )

    return report


def visualize_umap(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """Visualize embeddings using UMAP dimensionality reduction.

    Args:
        embeddings: shape (N, dim)
        labels: shape (N,)
        output_path: optional path to save the figure.
        figsize: figure size (width, height).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not installed. Install with: pip install umap-learn")
        return

    # Reduce to 2D using UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings.numpy())

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = torch.unique(labels).numpy()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=f"Class {label}",
            s=50,
            alpha=0.6,
            color=[color],
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Embedding Space Visualization (UMAP)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved UMAP visualization to {output_path}")

    plt.show()


def visualize_confusion_matrix(
    cm: np.ndarray,
    output_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Visualize confusion matrix as heatmap.

    Args:
        cm: confusion matrix of shape (num_classes, num_classes).
        output_path: optional path to save the figure.
        figsize: figure size (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Nearest Neighbor Classification)")

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {output_path}")

    plt.show()


def visualize_pca(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    n_components: int = 2,
) -> None:
    """Visualize embeddings using PCA dimensionality reduction.

    Args:
        embeddings: shape (N, dim)
        labels: shape (N,)
        output_path: optional path to save the figure.
        figsize: figure size (width, height).
        n_components: number of PCA components (2 or 3).
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3")

    # Reduce to n_components using PCA
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings.numpy())

    unique_labels = torch.unique(labels).numpy()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                embeddings_reduced[mask, 0],
                embeddings_reduced[mask, 1],
                label=f"Class {label}",
                s=50,
                alpha=0.6,
                color=[color],
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Embedding Space Visualization (PCA)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    else:  # n_components == 3
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                embeddings_reduced[mask, 0],
                embeddings_reduced[mask, 1],
                embeddings_reduced[mask, 2],
                label=f"Class {label}",
                s=50,
                alpha=0.6,
                color=[color],
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.set_title("Embedding Space Visualization (PCA)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved PCA visualization to {output_path}")

    plt.show()
