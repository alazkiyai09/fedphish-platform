"""Model architectures."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron for phishing detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class LogisticRegression(nn.Module):
    """Logistic regression for phishing detection."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        """
        Initialize logistic regression.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """Forward pass."""
        return self.linear(x)


class SimpleCNN(nn.Module):
    """Simple CNN for image-based phishing detection."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize CNN.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Forward pass."""
        # Assume input is 32x32
        x = x.view(-1, 1, 32, 32)
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(
    model_type: str = "mlp",
    input_dim: int = 100,
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Create model instance.

    Args:
        model_type: Type of model ("mlp", "logistic_regression", "simple_cnn")
        input_dim: Input dimension
        num_classes: Number of classes
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    if model_type == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dims=kwargs.get("hidden_dims", [128, 64]),
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            input_dim=input_dim,
            num_classes=num_classes,
        )
    elif model_type == "simple_cnn":
        return SimpleCNN(
            input_channels=kwargs.get("input_channels", 1),
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
