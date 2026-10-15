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
        # TODO: Initialize the layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        pass