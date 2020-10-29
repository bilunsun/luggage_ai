import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)

        self.backbone.to(device)
    
    def forward(self, x):
        return self.backbone(x)
    
    def training_step(self, batch, batch_index):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)