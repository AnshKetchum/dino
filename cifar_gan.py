'''
General goals of this code: 

    1 - Create an adaptable, general pipeline that enables devs to be able to understand what the
    convnet is seeing (through class activation maps like GradCAM)

    2 - Ensure the pipeline is **configurable** 

    The pipeline will leverage torch + torchvision + lightning + mlflow (logging)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping

import mlflow
import mlflow.pytorch

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tvf
from torchvision.utils import make_grid

# Grad-CAM imports -- useful for visualizing CNNs and Transformers
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Cool progress bars
from tqdm import tqdm
import numpy as np
import random

# Image manipulations
from PIL import Image

# Setup a basic transformation

# We'll keep it simple, for now
train_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transforms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# For this example, we'll use torchvision's CIFAR-10 example
train_dataset = torchvision.datasets.CIFAR10(
    './data', train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=7)

# For this example, we'll use torchvision's CIFAR-10 example
val_dataset = torchvision.datasets.CIFAR10(
    './data', train=False, download=True, transform=transforms)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=7)

# Define a model - this is the model I submitted for a CS 189 assignment, with a target of >75% val acc

# Define the discriminator


class ConvDiscriminator(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.mp = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):

        # First block, VGG inspired
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(self.mp(x))

        # Second block, VGG inspired
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(self.mp(x))

        # Third block, VGG inspired

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(self.mp(x))

        x = self.flat(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Turn into probability
        x = F.sigmoid(x)

        return x


class ConvGenerator(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ConvGenerator, self).__init__()

        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.78),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class LightningConv(L.LightningModule):

    def __init__(self, discriminator, generator, rand_num=100, embedding_dim=512) -> None:
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.embedding_dim = embedding_dim

        self.rand_num = rand_num
        self.i = 0

        self.n = 0
        self.generator_losses = []
        self.discriminator_losses = []

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def on_train_epoch_start(self):
        self.i += 1
        self.generator_losses = []
        self.discriminator_losses = []
        self.n = 0

    def on_train_epoch_end(self):
        mlflow.log_metric("train_generator_loss", sum(
            self.generator_losses), step=self.i)
        mlflow.log_metric("train_discriminator_loss", sum(
            self.discriminator_losses), step=self.i)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        return g_opt, d_opt

    def generate(self, batch_size):
        rand_vector = torch.randn(
            (batch_size, self.generator.embedding_dim), device=self.device)
        return self.generator(rand_vector)

    def training_step(self, batch, batch_idx):
        X_real, y_real = batch
        batch_size = X_real.shape[0]

        generator_optim, discriminator_optim = self.optimizers()

        generator_optim.zero_grad()
        discriminator_optim.zero_grad()

        # Optimize the discriminator

        disc_pred = self.discriminator(X_real)
        real_image_loss = F.binary_cross_entropy(disc_pred, torch.ones(
            (batch_size, 1), device=self.device))  # Discriminator Loss

        generated_image = self.generate(batch_size)

        disc_fake_pred = self.discriminator(generated_image)
        fake_image_loss = F.binary_cross_entropy(
            disc_fake_pred, torch.zeros((batch_size, 1), device=self.device))

        discriminator_loss = real_image_loss + fake_image_loss

        self.manual_backward(discriminator_loss)
        discriminator_optim.step()

        # Optimize the generator
        generated_image2 = self.generate(batch_size)
        generator_loss = F.binary_cross_entropy(self.discriminator(
            generated_image2), torch.ones((batch_size, 1), device=self.device))

        self.manual_backward(generator_loss)
        generator_optim.step()

        self.n += batch_size
        self.generator_losses.append(generator_loss)
        self.discriminator_losses.append(discriminator_loss)

        self.log("generator_loss", generator_loss, on_step=True, prog_bar=True)
        if batch_idx == self.rand_num:
            mlflow.log_image(tvf.to_pil_image(make_grid(generated_image2)),
                             f"train_epoch_{self.i}_batch_{batch_idx}_output.png", )


# View the training run live by typing 'mlflow ui'
experiment = mlflow.set_experiment("cifar_gan")
with mlflow.start_run(run_name="lightning_run", experiment_id=experiment.experiment_id):

    EMBEDDING_DIM = 1024

    # Hyperparameters
    latent_dim = 100
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    num_epochs = 10

    generator = ConvGenerator(EMBEDDING_DIM)
    discriminator = ConvDiscriminator(EMBEDDING_DIM)

    conv_lightning_model = LightningConv(
        discriminator, generator, rand_num=100, embedding_dim=EMBEDDING_DIM)
    # [EarlyStopping(monitor="val_accuracy", mode="min", patience=3)]
    callbacks = []

    trainer = L.Trainer(accelerator="cuda",
                        max_epochs=num_epochs, callbacks=callbacks)

    # Fit the lightning module to the dataset
    trainer.fit(conv_lightning_model, train_loader, val_loader)

    trainer.save_checkpoint('final', weights_only=True)
