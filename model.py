"""Embedding models for metric learning on face expression data."""

import timm
import torch
import torch.nn as nn


class FaceExpressionRecognitionModel(nn.Module):
    """Classification model for 7-class face expression recognition."""

    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, num_classes: int = 7):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class EmbeddingModel(nn.Module):
    """Embedding model for metric learning on face data.

    Design Rationale:
    ================
    We use a ResNet backbone (specifically resnet50 for good balance between
    capacity and inference speed) with a projection head for several reasons:

    1. **Backbone Choice (ResNet50)**:
       - Pre-trained on ImageNet, transfers well to facial features
       - Good capacity for learning discriminative embeddings
       - Efficient inference and training compared to larger models

    2. **Projection Head (512 -> embedding_dim)**:
       - Maps high-dimensional backbone features to embedding space
       - Intermediate layer (512) helps smooth feature transition
       - ReLU non-linearity preserves discriminative power
       - Final L2 normalization enables cosine distance computation

    3. **Why Metric Learning?**:
       - Instead of classification, we learn to pull same-class embeddings close
         and push different-class embeddings apart
       - Better generalization to unseen expressions
       - Enables efficient nearest-neighbor retrieval at inference
       - Works well with limited per-class samples (typical in RAFDB)

    4. **Embedding Dimension (default 128)**:
       - Balances computational efficiency with expressiveness
       - 128-dim is standard in metric learning literature
       - Reduces storage and distance computation cost vs higher dims

    5. **One-Stage Learning (Optional)**:
       - Can optionally add a parallel classifier head for combined
         Cross-Entropy and Contrastive learning.
       - Supports gating mechanism to weight features for each head.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        embedding_dim: int = 128,
        pretrained: bool = True,
        normalize: bool = True,
        num_classes: int = 0,
        use_gating: bool = False,
    ):
        """Initialize embedding model.

        Args:
            model_name: timm model name (e.g., "resnet50", "resnet18").
            embedding_dim: output embedding dimension.
            pretrained: use ImageNet pre-trained backbone.
            normalize: whether to L2-normalize embeddings.
            num_classes: if > 0, adds a parallel classifier head.
            use_gating: if True, adds gating mechanism for feature weighting.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.num_classes = num_classes
        self.use_gating = use_gating

        # Load backbone using timm
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )  # num_classes=0 removes classification head

        # Get backbone feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.backbone_dim = features.shape[1]

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

        # Optional: Classifier Head
        if self.num_classes > 0:
            self.classifier_head = nn.Linear(self.backbone_dim, num_classes)

        # Optional: Gating Mechanism
        if self.use_gating:
            self.gate_projection = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.Sigmoid()
            )
            if self.num_classes > 0:
                self.gate_classifier = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.Sigmoid()
                )

    def freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Also freeze the first layer of the projection head if you want to be stricter
        # for param in self.projection_head[0].parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Compute embeddings for input images.

        Args:
            x: input tensor of shape (B, 3, 224, 224).

        Returns:
            If num_classes=0: embeddings of shape (B, embedding_dim).
            If num_classes>0: tuple (embeddings, logits).
        """
        # Backbone: (B, 3, 224, 224) -> (B, backbone_dim)
        features = self.backbone(x)

        # Apply gating if enabled
        features_proj = features
        features_class = features

        if self.use_gating:
            features_proj = features * self.gate_projection(features)
            if self.num_classes > 0:
                features_class = features * self.gate_classifier(features)

        # Projection head: (B, backbone_dim) -> (B, embedding_dim)
        embeddings = self.projection_head(features_proj)

        # L2 normalization for cosine similarity
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if self.num_classes > 0:
            logits = self.classifier_head(features_class)
            return embeddings, logits

        return embeddings


class LinearClassifier(nn.Module):
    """A linear classifier to place on top of a frozen backbone."""

    def __init__(self, embedding_model: EmbeddingModel, num_classes: int = 7):
        """
        Args:
            embedding_model: The trained and frozen embedding model.
            num_classes: The number of output classes.
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.embedding_model.freeze_backbone()

        # New linear classifier head
        self.classifier = nn.Linear(embedding_model.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
        Returns:
            Logits of shape (B, num_classes).
        """
        with torch.no_grad():
            embeddings = self.embedding_model(x)
        return self.classifier(embeddings)


def validate_model_and_augmentation(
    model: nn.Module,
    dataset,
    num_samples: int = 4,
    device: torch.device = torch.device("cpu"),
) -> bool:
    """Validate that augmentation exists and model processes batches correctly.

    Args:
        model: embedding model to validate.
        dataset: dataset with augmentations.
        num_samples: number of samples to test.
        device: device to run on.

    Returns:
        True if validation passes, False otherwise.
    """
    model.eval()
    model.to(device)

    print("\n" + "=" * 70)
    print("VALIDATING MODEL AND AUGMENTATION")
    print("=" * 70)

    try:
        # Test 1: Check augmentation exists
        print("\n[Test 1] Checking augmentation...")
        sample_views, sample_label = dataset[0]

        if isinstance(sample_views, torch.Tensor):
            num_views = sample_views.shape[0]
            print(f"  ✓ Augmentation detected: {num_views} views per sample")
            print(f"    Each view shape: {tuple(sample_views.shape[1:])}")
        else:
            print("  ✗ Augmentation not found or incorrect format")
            return False

        # Test 2: Check input shape
        print("\n[Test 2] Checking input shapes...")
        expected_shape = (3, 224, 224)
        actual_shape = tuple(sample_views.shape[1:])

        if actual_shape == expected_shape:
            print(f"  ✓ Input shape correct: {actual_shape}")
        else:
            print(
                f"  ✗ Input shape mismatch. Expected {expected_shape}, got {actual_shape}"
            )
            return False

        # Test 3: Test batch processing
        print("\n[Test 3] Testing batch processing...")
        from torch.utils.data import DataLoader

        test_loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
        batch_images, batch_labels = next(iter(test_loader))

        print(f"  Batch size: {batch_images.shape[0]}")
        print(f"  Batch shape: {batch_images.shape}")

        # Expected: (B, C, H, W) after collate
        if len(batch_images.shape) == 4 and batch_images.shape[1] == 3:
            print("  ✓ Batch format correct: (B, C, H, W)")
        else:
            print("  ✗ Batch format incorrect. Expected (B, 3, 224, 224)")
            return False

        # Test 4: Forward pass
        print("\n[Test 4] Running forward pass...")
        batch_images = batch_images.to(device)
        with torch.no_grad():
            embeddings = model(batch_images)

        print(f"  Input shape:  {batch_images.shape}")
        print(f"  Output shape: {embeddings.shape}")

        if embeddings.shape == (batch_images.shape[0], model.embedding_dim):
            print(f"  ✓ Output shape correct: (B, {model.embedding_dim})")
        else:
            print(
                f"  ✗ Output shape mismatch. Expected (B, {model.embedding_dim}), "
                f"got {embeddings.shape}"
            )
            return False

        # Test 5: Check embedding statistics
        print("\n[Test 5] Checking embedding statistics...")
        norm = torch.norm(embeddings, p=2, dim=1)
        print(f"  L2 norm (should be ~1.0): {norm.mean():.4f} ± {norm.std():.4f}")

        if 0.95 < norm.mean() < 1.05:
            print("  ✓ Embeddings are properly normalized")
        else:
            print("  ⚠ Embeddings not normalized as expected")

        # Test 6: Check value ranges
        print("\n[Test 6] Checking value ranges...")
        print(f"  Input image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
        print(f"  Embedding range:   [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print(f"  Embedding mean:    {embeddings.mean():.3f}")
        print(f"  Embedding std:     {embeddings.std():.3f}")

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED - Ready to train!")
        print("=" * 70 + "\n")
        return True

    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Creating embedding model...")
    model = EmbeddingModel(model_name="resnet50", embedding_dim=128, pretrained=False)
    # print(f"Model:\n{model}\n")

    # Test with dummy input
    print("Testing with dummy input (B=4, C=3, H=224, W=224)...")
    dummy_input = torch.randn(4, 3, 224, 224)
    embeddings = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Output L2 norm: {torch.norm(embeddings, p=2, dim=1).mean():.4f}")
    