from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms.functional as tvf
import torchvision.transforms as T

# GaussianBlur in DINO augs isn't used.
from dino_augs import GaussianBlur, DINOCrops
from dino_loss import DINOLoss

import mlflow
import mlflow.pytorch

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from vit import vit_small, vit_tiny


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * \
        (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train(net: nn.Module, data: DataLoader, device: torch.device, epochs: int = 350):

    # Define our student and teacher networks
    student = net.to(device)
    teacher = deepcopy(net).to(device)

    # Freeze teacher weights
    for param in teacher.parameters():
        param.requires_grad = False

    # Define our optimizer
    optimizer = torch.optim.AdamW(student.parameters())

    # Define our loss
    warmup_teacher_temperature = 0.04
    teacher_temperature = 0.04
    warmup_teacher_temp_epochs = 0

    print(warmup_teacher_temperature,
          teacher_temperature, warmup_teacher_temp_epochs, epochs)

    # def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
    #              warmup_teacher_temp_epochs, nepochs, student_temperature=0.1, center_momentum=0.9, n_local_crops=8):

    criterion = DINOLoss(
        student.embed_dim,
        warmup_teacher_temperature,
        teacher_temperature,
        warmup_teacher_temp_epochs,
        epochs,
    ).to(device)

    # Define our teacher network momentum update schedule
    momentum_teacher = 0.996
    momentum_schedule = cosine_scheduler(momentum_teacher, 1,
                                         epochs, len(data))

    for epoch in range(epochs):

        total_loss = 0.0
        with tqdm(enumerate(data), unit="batch", total=len(data)) as tepoch:
            for it, (imgs, _) in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                local_crops, global_crops = imgs

                # Bring images over to our device
                local_crops = [l.to(device) for l in local_crops]
                global_crops = [g.to(device) for g in global_crops]

                # Run the student network through the LOCAL and GLOBAL crops
                student_embeddings_local = [student(l) for l in local_crops]
                student_embeddings_global = [student(g) for g in global_crops]

                # Run through the teacher embeddings on GLOBAL crops
                with torch.no_grad():
                    teacher_embeddings_global = [
                        teacher(g) for g in global_crops]

                # Compute the loss
                loss = criterion(torch.stack([*student_embeddings_local, *student_embeddings_global]),
                                 torch.stack(teacher_embeddings_global), epoch)

                total_loss += loss.item()
                loss.backward()

                optimizer.step()

                # Update the teacher network's weights using an EMA
                # EMA update for the teacher
                with torch.no_grad():
                    m = momentum_schedule[it]  # momentum parameter
                    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                        param_k.data.mul_(m).add_(
                            (1 - m) * param_q.detach().data)

                mlflow.log_metric(f"epoch_{epoch}_loss", loss.item())

                tepoch.set_postfix(loss=loss.item())

            print(f"Epoch Loss: {total_loss:.4f}")
            mlflow.log_metric(f"epoch_loss", total_loss.item())


'''
Training Notes: 

Batch Size: 64 


Rep. Collapse --> loss decreases, increases, stays constant at that high value

    Detect 
    - train and validate, 
        - get embedding matrix, grab matrix rank --> rank <= 20% * d, representational collapse
    
    Fix Collapse 
    - DINO --> 

'''

# Load in our data
with mlflow.start_run(run_name="dino_basic"):
    BATCH_SIZE = 16

    # Define our augmentations
    data_augs = DINOCrops(local_crops_scale=(0.05, 0.4),
                          global_crops_scale=(0.4, 1.))

    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=True, download=True, transform=data_augs)

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train a small vision transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = vit_small()

    print(net.embed_dim)
    # Train model using dino
    train(
        net,
        dataloader,
        device,
        epochs=5
    )
