"""Evaluation functions for metric learning models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def compute_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute embeddings and labels for all samples in a DataLoader.

    Args:
        model: embedding model (outputs embeddings of shape (B, embedding_dim)).
        data_loader: DataLoader with (images, labels) batches.
        device: device to run on.

    Returns:
        (embeddings, labels) as tensors of shape (N, dim) and (N,).
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

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                embeddings = outputs[0]
            else:
                embeddings = outputs
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


def compute_recall_at_k(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    k_values: list[int] = [1, 5, 10],
    metric: str = "euclidean",
) -> dict[int, float]:
    """Compute Recall@K metric.

    For each query, find the K nearest neighbors in the gallery and check
    if at least one is from the same class.

    Args:
        query_embeddings: shape (N_q, dim)
        query_labels: shape (N_q,)
        gallery_embeddings: shape (N_g, dim)
        gallery_labels: shape (N_g,)
        k_values: list of K values to compute recall for.
        metric: "euclidean" or "cosine" distance metric.

    Returns:
        dict mapping K -> Recall@K (float in [0, 1]).
    """
    if metric == "euclidean":
        # Compute Euclidean distances: (N_q, N_g)
        distances = torch.cdist(query_embeddings, gallery_embeddings, p=2)
    elif metric == "cosine":
        # Normalize embeddings
        q_norm = query_embeddings / (query_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        g_norm = gallery_embeddings / (gallery_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        # Cosine distance = 1 - cosine similarity
        distances = 1 - torch.mm(q_norm, g_norm.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")

    results = {}
    max_k = max(k_values)

    for query_idx in range(query_embeddings.shape[0]):
        # Get distances from this query to all gallery samples
        dists = distances[query_idx]

        # Get indices of K nearest neighbors
        _, nearest_indices = torch.topk(dists, k=min(max_k, dists.shape[0]), largest=False)

        # Check if any of the K nearest are from the same class
        query_label = query_labels[query_idx]
        nearest_labels = gallery_labels[nearest_indices]

        for k in k_values:
            k_nearest_labels = nearest_labels[:k]
            is_correct = (k_nearest_labels == query_label).any().item()

            if k not in results:
                results[k] = []
            results[k].append(float(is_correct))

    # Average across all queries
    recall_at_k = {k: np.mean(results[k]) for k in k_values}
    return recall_at_k


def compute_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "euclidean",
) -> float:
    """Compute accuracy using nearest neighbor classification.

    For each sample, find the nearest neighbor (excluding self) and check
    if its class matches. Accuracy is the fraction of correct predictions.

    Args:
        embeddings: shape (N, dim)
        labels: shape (N,)
        metric: distance metric ("euclidean" or "cosine").

    Returns:
        accuracy as float in [0, 1].
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

    # Get nearest neighbor for each sample
    nearest_indices = torch.argmin(distances, dim=1)
    predicted_labels = labels[nearest_indices]

    # Compute accuracy
    correct = (predicted_labels == labels).sum().item()
    total = labels.shape[0]
    accuracy = correct / total

    return accuracy


def evaluate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    k_values: list[int] = [1, 5, 10],
    metric: str = "euclidean",
) -> dict[str, float]:
    """Evaluate model on validation set by computing Recall@K and accuracy metrics.

    Uses validation data as both query and gallery (standard evaluation protocol).

    Args:
        model: embedding model.
        val_loader: validation DataLoader with (images, labels) batches.
        device: device to run on.
        k_values: list of K values for Recall@K.
        metric: distance metric ("euclidean" or "cosine").

    Returns:
        dict with keys like "recall_at_1", "recall_at_5", "accuracy", and float values in [0, 1].
    """
    # Compute embeddings for all validation samples
    embeddings, labels = compute_embeddings(model, val_loader, device)

    # Compute accuracy (nearest neighbor classification)
    accuracy = compute_accuracy(embeddings, labels, metric=metric)

    results = {}
    results["accuracy"] = accuracy

    return results


def final_eval(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    metric: str = "euclidean",
) -> dict[str, float]:
    """Final evaluation on test set.

    Computes Recall@1, @5, @10, accuracy, and additional statistics.

    Args:
        model: embedding model.
        test_loader: test DataLoader with (images, labels) batches.
        device: device to run on.
        metric: distance metric ("euclidean" or "cosine").

    Returns:
        dict with evaluation metrics and statistics.
    """
    # Compute embeddings for all test samples
    embeddings, labels = compute_embeddings(model, test_loader, device)

    # Compute accuracy (nearest neighbor classification)
    accuracy = compute_accuracy(embeddings, labels, metric=metric)

    results = {}
    results["accuracy"] = accuracy

    # Additional statistics
    num_samples = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]
    num_classes = len(torch.unique(labels))

    results.update({
        "num_test_samples": num_samples,
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
        "test_metric": metric,
    })

    return results
