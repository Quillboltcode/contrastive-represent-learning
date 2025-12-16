"""Training loop with pytorch-metric-learning miners and samplers."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_metric_learning.miners import BaseMiner
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device = torch.device("cpu"),
    miner: BaseMiner | None = None,
    epoch: int = 0,
    log_interval: int = 10,
) -> float:
    """Train for one epoch using a miner and optional triplet loss.

    Args:
        model: the embedding model (should output embeddings of shape (B, embedding_dim)).
        train_loader: DataLoader with (images, labels) or (embeddings, labels) batches.
        optimizer: optimizer for model parameters.
        loss_fn: loss function. If using a miner, typically TripletMarginLoss.
        device: device to run training on (cpu or cuda).
        miner: optional BaseMiner (e.g., TripletMarginMiner) to select hard triplets.
        epoch: epoch number for logging.
        log_interval: log every N batches.

    Returns:
        average loss over the epoch.
    """
    model.train()
    device = torch.device(device) if isinstance(device, str) else device
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        # Unpack batch: (images/embeddings, labels)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
        else:
            raise ValueError("Expected batch to be (inputs, labels) tuple or list")

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass: get embeddings
        embeddings = model(inputs)

        # If a miner is provided, use it to select hard triplets
        if miner is not None:
            # Miner returns anchor, positive, negative indices
            anchor_idx, positive_idx, negative_idx = miner(embeddings, labels)

            # If no triplets were mined, skip this batch
            if anchor_idx.numel() == 0:
                continue

            # Compute loss only on mined triplets
            loss = loss_fn(embeddings, labels, (anchor_idx, positive_idx, negative_idx))
        else:
            # Compute loss on the entire batch (typical for standard triplet loss)
            loss = loss_fn(embeddings, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar with current loss
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Return average loss for the epoch
    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_epoch_loss


def train_classifier_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device = torch.device("cpu"),
    epoch: int = 0,
    log_interval: int = 10,
) -> float:
    """Train a classifier for one epoch.

    Args:
        model: The classifier model.
        train_loader: DataLoader with (images, labels) batches.
        optimizer: optimizer for model parameters.
        loss_fn: loss function (e.g., CrossEntropyLoss).
        device: device to run training on.
        epoch: epoch number for logging.
        log_interval: log every N batches.

    Returns:
        average loss over the epoch.
    """
    model.train()
    device = torch.device(device) if isinstance(device, str) else device
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Classifier Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return total_loss / num_batches if num_batches > 0 else 0.0
