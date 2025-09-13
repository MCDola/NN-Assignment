import torch
import torch.optim as optim
from torch.nn import functional as F
from .model import VAE, AE
from .utils import get_dataloaders, plot_losses
from . import config

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def train_vae():
    train_loader, test_loader = get_dataloaders(config.batch_size, config.data_path)
    model = VAE(latent_dim=config.latent_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    train_losses = []
    test_losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar, config.beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % config.log_interval == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}')
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(config.device)
                recon_batch, mu, logvar = model(data)
                test_loss += vae_loss(recon_batch, data, mu, logvar, config.beta).item()
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        print(f'Test Loss: {avg_test_loss:.4f}')

    torch.save(model.state_dict(), f'{config.model_save_path}/vae_model.pth')
    plot_losses(train_losses, test_losses, f'{config.figure_save_path}/vae_losses.png')

def train_ae():
    train_loader, test_loader = get_dataloaders(config.batch_size, config.data_path)
    model = AE(latent_dim=config.latent_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    train_losses = []
    test_losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config.device)
            optimizer.zero_grad()
            recon_batch = model(data)
            loss = F.mse_loss(recon_batch, data, reduction='sum')
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % config.log_interval == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}')
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(config.device)
                recon_batch = model(data)
                test_loss += F.mse_loss(recon_batch, data, reduction='sum').item()
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        print(f'Test Loss: {avg_test_loss:.4f}')

    torch.save(model.state_dict(), f'{config.model_save_path}/ae_model.pth')
    plot_losses(train_losses, test_losses, f'{config.figure_save_path}/ae_losses.png')