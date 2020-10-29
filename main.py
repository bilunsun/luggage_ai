import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from dataset_utils import get_loaders
from img_utils import show_random_images
from neural_network import Model


def main(path):
    train_loader, test_loader = get_loaders(path)

    model = Model()

    trainer = pl.Trainer(gpus=1, max_epochs=3)

    trainer.fit(model, train_loader, test_loader)

    torch.save(model.state_dict(), "backbone.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",
                        help="Folder containing the luggage images",
                        required=True,
                        type=str)

    args = parser.parse_args()

    main(args.folder)