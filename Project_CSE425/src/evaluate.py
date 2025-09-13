import torch
from .model import VAE, AE
from .utils import get_dataloaders, save_images
from . import config

def generate_samples_vae():
    model = VAE(latent_dim=config.latent_dim).to(config.device)
    model.load_state_dict(torch.load(f'{config.model_save_path}/vae_model.pth'))
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, config.latent_dim).to(config.device)
        samples = model.decode(z)
        save_images(samples, f'{config.figure_save_path}/vae_samples.png')

def generate_samples_ae():
    model = AE(latent_dim=config.latent_dim).to(config.device)
    model.load_state_dict(torch.load(f'{config.model_save_path}/ae_model.pth'))
    model.eval()
    _, test_loader = get_dataloaders(config.batch_size, config.data_path)
    data, _ = next(iter(test_loader))
    data = data.to(config.device)
    with torch.no_grad():
        recon = model(data)
    save_images(recon, f'{config.figure_save_path}/ae_reconstructions.png')