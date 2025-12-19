"""Main training script for RAFDB with metric learning, augmentation, and wandb logging."""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

import wandb
from tqdm import tqdm
from pytorch_metric_learning.losses import SupConLoss, TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.samplers import MPerClassSampler

from dataset import train_val_split
from train import train_one_epoch, train_classifier_one_epoch
from evalute import evaluate_one_epoch, final_eval
from visualize import (
    compute_embeddings_and_predictions,
    compute_confusion_matrix,
    compute_classification_report,
    visualize_confusion_matrix,
    visualize_pca,
)
from model import EmbeddingModel, LinearClassifier, validate_model_and_augmentation


class RAFDBWithAugmentation(Dataset):
    """RAFDB dataset wrapper with controlled augmentation variance.

    For each sample, creates multiple augmented views with different
    transformations: crop, rotate, grayscale.
    """

    def __init__(
        self,
        root: str,
        split: str = "train", # "train" or "test"
        num_augmentations: int = 2,
        crop_scale: tuple[float, float] = (0.8, 1.0),
        rotation_degrees: int = 15,
        grayscale_prob: float = 0.3,
        use_autoaugment: bool = False,
    ):
        """Initialize RAFDB dataset with augmentation options.

        Args:
            root: path to RAFDB directory.
            split: "train" or "test".
            num_augmentations: number of augmented views per image.
            crop_scale: (min_scale, max_scale) for random crop.
            rotation_degrees: max rotation in degrees.
            grayscale_prob: probability of grayscale augmentation.
            use_autoaugment: whether to use AutoAugment.
        """
        self.root = Path(root)
        self.split = split
        self.num_augmentations = num_augmentations
        self.crop_scale = crop_scale
        self.rotation_degrees = rotation_degrees
        self.grayscale_prob = grayscale_prob
        self.use_autoaugment = use_autoaugment

        # Load RAFDB dataset (assumes standard RAFDB structure)
        # RAFDB: train/test folders with emotion subfolders
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"RAFDB split directory not found: {split_dir}")

        # Base transform: resize to 224x224
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Augmentation transforms with variance
        self.augmentation_transforms = [
            self._create_augmentation_transform()
            for _ in range(num_augmentations)
        ]

        # Load images and labels using ImageFolder
        self.dataset = datasets.ImageFolder(
            str(split_dir),
            transform=None,  # We'll apply transforms manually
        )

    def _create_augmentation_transform(self):
        """Create a single augmentation pipeline with variance."""
        if self.use_autoaugment:
            transforms_list = [
                transforms.RandomResizedCrop(224, scale=self.crop_scale),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            transforms_list = [
                transforms.RandomResizedCrop(224, scale=self.crop_scale),
                transforms.RandomRotation(self.rotation_degrees),
            ]
            if self.grayscale_prob > 0:
                transforms_list.append(transforms.RandomGrayscale(p=self.grayscale_prob))
            transforms_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get original image and label
        img, label = self.dataset[idx]
        
        if self.split == 'train' or self.num_augmentations > 0:
            # For training, create multiple augmented views for contrastive learning
            img_augmented = [
                aug_transform(img) for aug_transform in self.augmentation_transforms
            ]
            
            # Stack all views: (num_augmentations, 3, 224, 224)
            all_views = torch.stack(img_augmented, dim=0)
        else:
            # For validation/test, just apply the base transform
            all_views = self.base_transform(img)

        return all_views, label


def visualize_triplet_mining(
    model: nn.Module,
    train_loader: DataLoader,
    miner,
    device: torch.device,
    num_samples: int = 3,
):
    """Visualize triplet mining results to verify anchor/positive/negative selection.

    Args:
        model: embedding model.
        train_loader: training DataLoader.
        miner: TripletMarginMiner instance.
        device: device to run on.
        num_samples: number of triplets to visualize.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("⚠ matplotlib not installed. Skipping triplet visualization.")
        return

    model.eval()
    print("\n" + "=" * 70)
    print("VISUALIZING TRIPLET MINING RESULTS")
    print("=" * 70)

    with torch.no_grad():
        # Get first batch
        batch_images, batch_labels = next(iter(train_loader))
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        # Get embeddings
        embeddings = model(batch_images)

        # Mine triplets
        anchor_idx, positive_idx, negative_idx = miner(embeddings, batch_labels)

        if anchor_idx.numel() == 0:
            print("⚠ Miner returned 0 triplets. Skipping visualization.")
            return

        print(f"✓ Mined {anchor_idx.numel()} triplets from batch of {batch_images.shape[0]} images")

        # Visualize first num_samples triplets
        num_to_show = min(num_samples, anchor_idx.numel())

        for triplet_num in range(num_to_show):
            a_idx = anchor_idx[triplet_num].item()
            p_idx = positive_idx[triplet_num].item()
            n_idx = negative_idx[triplet_num].item()

            a_label = batch_labels[a_idx].item()
            p_label = batch_labels[p_idx].item()
            n_label = batch_labels[n_idx].item()

            # Get images
            a_img = batch_images[a_idx]  # (3, 224, 224)
            p_img = batch_images[p_idx]
            n_img = batch_images[n_idx]

            # Denormalize images (undo ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

            a_img_vis = (a_img * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            p_img_vis = (p_img * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            n_img_vis = (n_img * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()

            # Create figure
            fig = plt.figure(figsize=(14, 5))
            gs = GridSpec(1, 3, figure=fig, wspace=0.3)

            # Anchor
            ax_a = fig.add_subplot(gs[0, 0])
            ax_a.imshow(a_img_vis)
            ax_a.set_title(f"Anchor\nLabel: {a_label}", fontsize=12, fontweight="bold", color="blue")
            ax_a.axis("off")

            # Positive (same class as anchor)
            ax_p = fig.add_subplot(gs[0, 1])
            ax_p.imshow(p_img_vis)
            status = "✓ Same" if p_label == a_label else "✗ Wrong"
            ax_p.set_title(
                f"Positive\nLabel: {p_label}\n{status}",
                fontsize=12,
                fontweight="bold",
                color="green" if p_label == a_label else "red",
            )
            ax_p.axis("off")

            # Negative (different class from anchor)
            ax_n = fig.add_subplot(gs[0, 2])
            ax_n.imshow(n_img_vis)
            status = "✓ Different" if n_label != a_label else "✗ Wrong"
            ax_n.set_title(
                f"Negative\nLabel: {n_label}\n{status}",
                fontsize=12,
                fontweight="bold",
                color="green" if n_label != a_label else "red",
            )
            ax_n.axis("off")

            fig.suptitle(
                f"Triplet {triplet_num + 1}/{num_to_show}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.show()

            print(
                f"  Triplet {triplet_num + 1}: "
                f"Anchor(L={a_label}) -> Pos(L={p_label}) + Neg(L={n_label}) | "
                f"Valid: {p_label == a_label and n_label != a_label}"
            )

    print("=" * 70)


def train_combined_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_metric_fn: nn.Module,
    loss_ce_fn: nn.Module,
    alpha: float,
    device: torch.device,
    miner=None,
    epoch: int = 0,
    log_interval: int = 10,
) -> dict:
    """Train one epoch with combined Metric and Cross-Entropy loss."""
    model.train()
    total_loss = 0.0
    total_metric_loss = 0.0
    total_ce_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass: get embeddings and logits
        embeddings, logits = model(images)

        # 1. Metric Learning Loss
        if miner is not None:
            indices_tuple = miner(embeddings, labels)
            loss_metric = loss_metric_fn(embeddings, labels, indices_tuple)
        else:
            loss_metric = loss_metric_fn(embeddings, labels)

        # 2. Cross-Entropy Loss
        loss_ce = loss_ce_fn(logits, labels)

        # 3. Combined Loss
        loss = alpha * loss_metric + (1.0 - alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_metric_loss += loss_metric.item()
        total_ce_loss += loss_ce.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                "L_tot": f"{total_loss/num_batches:.3f}",
                "L_met": f"{total_metric_loss/num_batches:.3f}",
                "L_ce": f"{total_ce_loss/num_batches:.3f}"
            })

    return {
        "total": total_loss / num_batches if num_batches > 0 else 0.0,
        "metric": total_metric_loss / num_batches if num_batches > 0 else 0.0,
        "ce": total_ce_loss / num_batches if num_batches > 0 else 0.0,
    }

def collate_fn_with_augmentations(batch):
    """Custom collate to handle augmented views.

    Expects batch items of (stacked_views, label) where stacked_views
    is shape (num_aug+1, 3, 224, 224).

    Returns (images, labels) where images are flattened to (B, 3, 224, 224).
    """
    views_list = []
    labels_list = []

    for views, label in batch:
        # Flatten augmentations for this sample
        # views can be (num_aug, 3, 224, 224) or just (3, 224, 224)
        if views.dim() == 4:
            num_views = views.shape[0]
            for v in range(num_views):
                views_list.append(views[v])
                labels_list.append(label)
        else:
            # Single view (3, 224, 224)
            views_list.append(views)
            labels_list.append(label)

    images = torch.stack(views_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return images, labels


def setup_wandb(config: dict):
    """Initialize wandb for experiment tracking."""
    wandb.init(
        project="contrast-represent-learning",
        name=config.get("run_name", "rafdb-metric-learning"),
        config=config,
    )


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.debug:
        print("\n!!! DEBUG MODE ENABLED !!!")
        print("Overriding num_epochs to 2")
        args.num_epochs = 2

    print("\n" + "=" * 70)
    print("HYPERPARAMETERS")
    print("=" * 70)
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup wandb
    config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "embedding_dim": args.embedding_dim,
        "temperature": args.temperature,
        "metric": args.metric,
        "crop_scale": args.crop_scale,
        "rotation_degrees": args.rotation_degrees,
        "grayscale_prob": args.grayscale_prob,
        "use_autoaugment": args.use_autoaugment,
        "m_per_class": args.m_per_class,
        "loss_type": args.loss_type,
        "alpha": args.alpha,
        "use_gating": args.use_gating,
    }
    if args.use_wandb:
        setup_wandb(config)

    # Load RAFDB dataset
    print("Loading RAFDB dataset...")
    print(f"  RAFDB root: {args.rafdb_root}")
    print(f"  Augmentations: {args.num_augmentations} views per sample")
    print(f"  Crop scale: {args.crop_scale}")
    print(f"  Rotation: ±{args.rotation_degrees}°")
    print(f"  Grayscale prob: {args.grayscale_prob}")
    print(f"  Using AutoAugment: {args.use_autoaugment}")
    
    # We need to manually split indices to create proper Train (augmented) and Val (clean) sets
    # First, create a dummy dataset to get the split indices
    temp_dataset = datasets.ImageFolder(str(Path(args.rafdb_root) / "train"))
    train_subset, val_subset = train_val_split(
        temp_dataset, val_fraction=args.val_fraction, seed=args.seed
    )
    train_indices = train_subset.indices
    val_indices = val_subset.indices

    # Create the actual Train dataset with augmentations
    train_ds_full = RAFDBWithAugmentation(
        root=args.rafdb_root,
        split="train",
        num_augmentations=args.num_augmentations,
        crop_scale=tuple(args.crop_scale),
        rotation_degrees=args.rotation_degrees,
        grayscale_prob=args.grayscale_prob,
        use_autoaugment=args.use_autoaugment,    
    )
    train_ds = torch.utils.data.Subset(train_ds_full, train_indices)

    # Create the actual Val dataset (clean, single view)
    # We use split='train' to point to the folder, but num_augmentations=1 for single view
    val_ds_full = RAFDBWithAugmentation(
        root=args.rafdb_root, split="train", num_augmentations=1
    )
    val_ds = torch.utils.data.Subset(val_ds_full, val_indices)

    # Create samplers
    train_labels = torch.tensor([temp_dataset.targets[i] for i in train_indices])
    num_classes = len(torch.unique(train_labels))
    max_batch_size = args.m_per_class * num_classes

    print(f"\nDataLoader setup:")
    print(f"  Num unique classes in train set: {num_classes}")
    print(f"  M_PER_CLASS: {args.m_per_class}")
    print(f"  Max allowed batch_size: {max_batch_size}")
    print(f"  Requested batch_size: {args.batch_size}")

    if args.batch_size > max_batch_size:
        print(f"❌ ERROR: batch_size ({args.batch_size}) > m*num_classes ({max_batch_size})")
        print("MPerClassSampler requires: batch_size <= m * (number of unique labels)")
        print("\nFix options:")
        print(f"  1. Reduce batch_size to {max_batch_size} or less")
        print(f"  2. Increase M_PER_CLASS (e.g., from {args.m_per_class} to {int(args.batch_size/num_classes)+1})")
        print("  3. Keep original batch_size but use a different sampler (not MPerClassSampler)")
        if args.use_wandb:
            wandb.finish()
        raise AssertionError(f"batch_size ({args.batch_size}) must be <= m*num_classes ({max_batch_size})")

    sampler = MPerClassSampler(
        train_labels,
        m=args.m_per_class,
        length_before_new_iter=len(train_ds),
        batch_size=args.batch_size,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_augmentations,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_augmentations,
    )

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = RAFDBWithAugmentation(
        root=args.rafdb_root,
        split="test",
        num_augmentations=args.num_augmentations,
        crop_scale=tuple(args.crop_scale),
        rotation_degrees=args.rotation_degrees,
        grayscale_prob=args.grayscale_prob,
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_augmentations,
    )

    # Model, loss, optimizer
    print("\n" + "=" * 70)
    print("STEP 2: Initializing embedding model...")
    print("=" * 70)
    
    model = EmbeddingModel(
        model_name="resnet50",
        embedding_dim=args.embedding_dim,
        pretrained=True,
        normalize=True,
        num_classes=num_classes,  # Enable classifier head
        use_gating=args.use_gating,
    )
    model.to(device)
    print("✓ Model initialized:")
    print("  - Backbone: ResNet50 (pretrained)")
    print(f"  - Embedding dim: {args.embedding_dim}")
    print("  - Normalization: Enabled (L2)")
    print(f"  - One-Stage Learning: True (alpha={args.alpha})")
    print(f"  - Device: {device}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss setup
    criterion_ce = nn.CrossEntropyLoss()
    
    miner = None
    if args.loss_type == "supcon":
        print(f"  - Metric Loss: SupConLoss (temp={args.temperature})")
        criterion_metric = SupConLoss(temperature=args.temperature)
    else:
        print("  - Metric Loss: TripletMarginLoss (margin=0.2)")
        criterion_metric = TripletMarginLoss(margin=0.2)
        print("  - Miner: TripletMarginMiner")
        miner = TripletMarginMiner(margin=0.2, type_of_triplets="semihard")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=0.1
    )

    # This visualization is for triplets, so we'll skip it for SupCon
    if miner is not None:
        visualize_triplet_mining(model, train_loader, miner, device)
    else:
        print("\n" + "=" * 70)
        print("STEP 3A: Skipping triplet visualization (using SupConLoss)...")
        print("=" * 70)

    # Training loop
    print("\n" + "=" * 70)
    print("STEP 3B: Starting one-stage training...")
    print("=" * 70)
    best_accuracy = 0.0

    for epoch in range(args.num_epochs):
        # Train
        train_results = train_combined_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_metric_fn=criterion_metric,
            loss_ce_fn=criterion_ce,
            alpha=args.alpha,
            device=device,
            miner=miner,
            epoch=epoch,
        )
        train_loss = train_results["total"]

        # Validate
        val_metrics = evaluate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            metric=args.metric,
        )

        scheduler.step()

        # Log to wandb
        if args.use_wandb:
            log_dict = {
                "train_loss": train_loss,
                "train_loss_metric": train_results["metric"],
                "train_loss_ce": train_results["ce"],
            }
            log_dict.update(val_metrics)
            log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=epoch)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Recall@1: {val_metrics.get('recall_at_1', 0):.4f}"
        )

        # Save best model based on validation accuracy
        if val_metrics.get("accuracy", 0) > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            model_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")

    # Final evaluation and visualization
    print("Final evaluation and visualization...")
    embeddings, labels = compute_embeddings_and_predictions(
        model, val_loader, device=device
    )

    cm = compute_confusion_matrix(embeddings, labels, metric=args.metric)
    visualize_confusion_matrix(
        cm, output_path=str(output_dir / "confusion_matrix.png")
    )
    visualize_pca(embeddings, labels, output_path=str(output_dir / "pca.png"))

    # Final test evaluation
    print("\n" + "=" * 70)
    print("STEP 5: Final evaluation on test set...")
    print("=" * 70)
    test_metrics = final_eval(
        model, test_loader, device=device, metric=args.metric
    )
    print(f"Test Recall@1: {test_metrics.get('recall_at_1', 0):.4f}")
    print(f"Test Recall@5: {test_metrics.get('recall_at_5', 0):.4f}")
    print(f"Test Recall@10: {test_metrics.get('recall_at_10', 0):.4f}")
    print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")

    if args.use_wandb:
        wandb.log({
            "test_recall_at_1": test_metrics.get("recall_at_1", 0),
            "test_recall_at_5": test_metrics.get("recall_at_5", 0),
            "test_recall_at_10": test_metrics.get("recall_at_10", 0),
            "test_accuracy": test_metrics.get("accuracy", 0),
        })

    if args.use_wandb:
        wandb.finish()

    print(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train metric learning model on RAFDB with wandb logging."
    )

    # Dataset args
    parser.add_argument(
        "--rafdb_root", type=str, required=True, help="Path to RAFDB root directory"
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.2, help="Validation fraction"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Augmentation args
    parser.add_argument(
        "--num_augmentations", type=int, default=2, help="Number of augmented views"
    )
    parser.add_argument(
        "--use_autoaugment", action="store_true", help="Use AutoAugment"
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        nargs=2,
        default=[0.8, 1.0],
        help="Crop scale range",
    )
    parser.add_argument(
        "--rotation_degrees", type=int, default=15, help="Max rotation in degrees"
    )
    parser.add_argument(
        "--grayscale_prob", type=float, default=0.3, help="Grayscale probability"
    )

    # Training args
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--num_epochs_classifier", type=int, default=20, help="Number of epochs for linear classifier")
    parser.add_argument("--batch_size", type=int, default=28, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_step", type=int, default=10, help="LR scheduler step")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    # Model args
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )

    # Metric learning args
    parser.add_argument("--temperature", type=float, default=0.1, help="SupCon temperature")
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric",
    )
    parser.add_argument(
        "--m_per_class", type=int, default=4, help="Samples per class in batch"
    )
    parser.add_argument(
        "--loss_type", type=str, default="supcon", choices=["supcon", "triplet"], help="Type of metric loss"
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for metric loss (1-alpha for CE)")
    parser.add_argument("--use_gating", action="store_true", help="Use gating mechanism in model")


    # Logging/output args
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb logging"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (2 epochs)")

    args = parser.parse_args()
    main(args)
