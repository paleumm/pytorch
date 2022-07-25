import os
from statistics import mode
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        self.log("Training Loss : ", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

dataset = MNIST(root="./data", download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=dataset,num_workers=16)

autoencoder = LitAutoEncoder()

trainer = pl.Trainer(accelerator="gpu", devices=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

torch.save(autoencoder.state_dict(), "autoencoder_checkpoint.pth")