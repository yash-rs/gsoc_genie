import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, x0, v0, t_max, activation="tanh"):
        super().__init__()

        self.x_0 = x0
        self.v_0 = v0
        self.t_max = t_max

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
        Hard constrains ICs
        """
        t = input[:, 0:1]
        N = self.network(input)
        x = self.x_0 + (self.v_0 * self.t_max) * t + torch.pow(t, 2) * N # hardcode ICs
        return x

    def _initialize_weights(self):
        """
        Xavier initialization (important for PINNs).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

class FourierFCNN(nn.Module):
    def __init__(self,
                 hidden_dim,
                 hidden_layers,
                 x0,
                 v0,
                 t_max,
                 num_frequencies=10,
                 scale=5.0,
                 activation="tanh"):
        super().__init__()

        self.B = torch.randn(num_frequencies) * scale #fourier frequencies
        self.input_dim = 2 * (num_frequencies) + 1
        self.x_0 = x0
        self.v_0 = v0
        self.t_max = t_max
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
        layers.append(nn.Linear(self.input_dim, hidden_dim))
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
        t = input[:, 0:1]
        xi = input[:, 1:2]
        t_proj = 2.0 * torch.pi * t * self.B

        input = torch.cat(
            [torch.sin(t_proj), torch.cos(t_proj), xi],
            dim=1
        )
        N = self.network(input)
        x = self.x_0 + (self.v_0 * self.t_max) * t + torch.pow(t, 2) * N # hardcode ICs
        return x
    
    def _initialize_weights(self):
        """
        Xavier initialization (important for PINNs).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

