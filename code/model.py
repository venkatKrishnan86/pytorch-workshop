import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list[int], 
        activations: list[nn.Module],
        output_dim: int,
        output_activation: nn.Module | None = None
    ) -> None:
        super(MLP, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(activations[0])
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(activations[i])
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if output_activation is not None:
            self.layers.append(output_activation)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

if __name__ == "__main__":
    mlp = MLP(input_dim=256, hidden_dims=[20], activations=[nn.ReLU()], output_dim=5, output_activation=nn.Sigmoid())
    x = torch.randn(4, 256)
    print(mlp(x))