import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm


class CombinedDataset(Dataset):
    def __init__(self, path, transform):
        cirfar_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        luggage_dataset = datasets.ImageFolder(root=path, transform=transform)
        limit = len(luggage_dataset)

        self.images = []
        self.labels = []

        for i, (img, _) in tqdm(enumerate(cirfar_dataset)):
            if i == limit:
                break

            self.images.append(img)
            self.labels.append(1)

        for i, (img, _) in tqdm(enumerate(luggage_dataset)):
            if i == limit:
                break
            self.images.append(img)
            self.labels.append(0)
        
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def get_loaders(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((32, 32))
    ])

    dataset = CombinedDataset(path, transform)

    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset) - train_len

    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    return train_loader, test_loader
