import torch
import json
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model import VAE, AE
from src.utils import get_dataloaders, save_images
import src.config as config

def compute_metrics(model, dataloader, device, model_type='vae'):
    """Compute FID and IS scores"""
    fid = FrechetInceptionDistance(feature=64).to(device)
    inception = InceptionScore().to(device)
    
    real_images = []
    generated_images = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if i > 20: 
                break
                
            data = data.to(device)

            if model_type == 'vae':
                recon_batch, _, _ = model(data)
            else:
                recon_batch = model(data)

            real_batch_rgb = data.repeat(1, 3, 1, 1) 
            gen_batch_rgb = recon_batch.repeat(1, 3, 1, 1)  

            real_batch = (real_batch_rgb * 255).byte()
            gen_batch = (gen_batch_rgb * 255).byte()
            
            # Update metrics
            fid.update(real_batch, real=True)
            fid.update(gen_batch, real=False)
            inception.update(gen_batch)

            if len(real_images) < 32:
                real_images.append(data.cpu())
                generated_images.append(recon_batch.cpu())
    
    fid_score = fid.compute()
    is_mean, is_std = inception.compute()
    
    return (fid_score.item(), is_mean.item(), is_std.item(), 
            torch.cat(real_images), torch.cat(generated_images))

def run_complete_evaluation():
    """Run full evaluation and generate all outputs"""
    print("Loading test data...")
    _, test_loader = get_dataloaders(64, config.data_path)
    
    results = {}
    
    # Evaluate VAE
    print("Evaluating VAE...")
    vae_model = VAE(latent_dim=config.latent_dim).to(config.device)
    vae_model.load_state_dict(torch.load(f'{config.model_save_path}/vae_model.pth'))
    vae_model.eval()
    
    vae_fid, vae_is, vae_is_std, vae_real, vae_gen = compute_metrics(vae_model, test_loader, config.device, 'vae')
    results['VAE'] = {'FID': vae_fid, 'IS': vae_is, 'IS_std': vae_is_std}
    
    # Save VAE samples
    save_images(vae_real[:64], f'{config.figure_save_path}/vae_real_samples.png', nrow=8)
    save_images(vae_gen[:64], f'{config.figure_save_path}/vae_reconstructed_samples.png', nrow=8)
    
    # Evaluate AE
    print("Evaluating AE...")
    ae_model = AE(latent_dim=config.latent_dim).to(config.device)
    ae_model.load_state_dict(torch.load(f'{config.model_save_path}/ae_model.pth'))
    ae_model.eval()
    
    ae_fid, ae_is, ae_is_std, ae_real, ae_gen = compute_metrics(ae_model, test_loader, config.device, 'ae')
    results['AE'] = {'FID': ae_fid, 'IS': ae_is, 'IS_std': ae_is_std}
    
    # Save AE samples
    save_images(ae_real[:64], f'{config.figure_save_path}/ae_real_samples.png', nrow=8)
    save_images(ae_gen[:64], f'{config.figure_save_path}/ae_reconstructed_samples.png', nrow=8)
    
    # Generate random samples from VAE
    print("Generating random samples from VAE...")
    with torch.no_grad():
        z = torch.randn(64, config.latent_dim).to(config.device)
        random_samples = vae_model.decode(z)
    save_images(random_samples, f'{config.figure_save_path}/vae_random_samples.png', nrow=8)
    
    # Save results
    with open(f'{config.metrics_save_path}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"VAE - FID: {vae_fid:.2f}, IS: {vae_is:.2f} ± {vae_is_std:.2f}")
    print(f"AE  - FID: {ae_fid:.2f}, IS: {ae_is:.2f} ± {ae_is_std:.2f}")
    
    return results

def demonstrate_uncertainty(model, dataloader, device, num_samples=5):
    """Generate multiple reconstructions for the same input to show uncertainty"""
    model.eval()
    data, _ = next(iter(dataloader))
    data = data[:1].to(device) 
    
    reconstructions = []
    with torch.no_grad():
        for _ in range(num_samples):
            recon, _, _ = model(data) 
            reconstructions.append(recon.cpu())
    
    all_recon = torch.cat(reconstructions)
    save_images(all_recon, f'{config.figure_save_path}/uncertainty_demo.png', nrow=num_samples)

if __name__ == '__main__':
    run_complete_evaluation()