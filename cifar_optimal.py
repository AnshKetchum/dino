import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm

# Load in models
from models.encoder import ConvEncoder
from models.decoder import ConvDecoder

# Logging
import mlflow
import mlflow.pytorch


def train(net: nn.Module, dataloader: DataLoader, device: torch.device, epochs: int = 30):
    net.train()

    # Define an optimizer
    optim = torch.optim.Adam(net.parameters())

    for i in range(epochs):

        losses = []
        for j, (X, y) in tqdm(enumerate(dataloader)):

            X = X.to(device)
            y = y.to(device)

            pass

        mlflow.log_metric('train_loss', sum(losses), step=i)
