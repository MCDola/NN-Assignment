import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

def create_dirs():
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./outputs/models', exist_ok=True)
    os.makedirs('./outputs/figures', exist_ok=True)
    os.makedirs('./outputs/metrics', exist_ok=True)

def get_dataloaders(batch_size=128, data_path='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def save_images(images, path, nrow=8):
    save_image(images, path, nrow=nrow, normalize=True)

def plot_losses(train_losses, test_losses, path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()