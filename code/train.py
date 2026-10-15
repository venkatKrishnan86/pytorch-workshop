import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset import MoonsDataset
from model import MLP

torch.manual_seed(0)
np.random.seed(0)

NUM_EPOCHS = 1000

def train_epoch(
    train_loader: DataLoader, 
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: str
) -> float:
    """
        Must return the mean loss of the model on the given data loader.
    """
    total_loss = 0
    total_samples = 0
    model.train()
    
    for _, (X, y_true) in enumerate(train_loader):
        # X: (batch, 2)
        # y: (batch, 1)
        X = X.to(device)
        y_true = y_true.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.shape[0]
        total_samples += X.shape[0]

    if scheduler is not None:
        scheduler.step()

    return total_loss/total_samples

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> tuple[float, float]:
    """
        Must return the mean accuracy and loss of the model on the given data loader as a tuple.
    """
    model.eval()

    correct = 0
    losses = 0
    total = 0

    with torch.no_grad():
        for _, (X, y_true) in enumerate(dataloader):
            X = X.to(device)
            y_true = y_true.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)

            losses += loss.item() * X.shape[0]

            correct += (y_pred.round() == y_true).sum().item()
            total += X.shape[0]

    val_acc = correct/total
    val_loss = losses/total

    return val_acc, val_loss

        

if __name__ == "__main__":
    dataset = MoonsDataset(n_samples=1000, noise=0.2, random_state=0)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=0)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = MLP(
        input_dim=2, 
        hidden_dims=[10, 10], 
        activations=[nn.ReLU(), nn.ReLU()], 
        output_dim=1, 
        output_activation=nn.Sigmoid()
    )

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, scheduler, device)
        val_acc, val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}, Train Loss {train_loss}, Val Acc {val_acc}, Val Loss {val_loss}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/loss.png")
    plt.close()
