import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, activation="tanh"):
        super().__init__()

        # Activation selection
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act())

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, input):
        """
        Forward pass.
        Concatenates (t, xi) along feature dimension.
        """
        return self.network(input)

    def _initialize_weights(self):
        """
        Xavier initialization (important for PINNs).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)