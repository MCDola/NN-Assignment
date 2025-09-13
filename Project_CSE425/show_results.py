import json
import matplotlib.pyplot as plt
from PIL import Image
import os

def display_results():
    """Display all results in a structured way"""
    
    # Load evaluation results
    with open('outputs/metrics/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print("=" * 50)
    print("NEURAL NETWORKS COURSE - ASSIGNMENT RESULTS")
    print("=" * 50)
    
    print("\n1. QUANTITATIVE EVALUATION METRICS:")
    print("-" * 40)
    for model, metrics in results.items():
        print(f"{model}:")
        print(f"  FID Score: {metrics['FID']:.2f} (lower is better)")
        print(f"  Inception Score: {metrics['IS']:.2f} ± {metrics['IS_std']:.2f} (higher is better)")
        print()
    
    print("\n2. QUALITATIVE RESULTS (check outputs/figures folder):")
    print("-" * 40)
    figures = [
        "vae_real_samples.png - Original test images",
        "vae_reconstructed_samples.png - VAE reconstructions", 
        "ae_reconstructed_samples.png - AE reconstructions",
        "vae_random_samples.png - Randomly generated samples from VAE"
    ]
    
    for fig in figures:
        print(f"  • {fig}")
    
    print("\n3. INTERPRETATION:")
    print("-" * 40)
    print("• FID measures similarity between real and generated images")
    print("• IS measures diversity and quality of generated images")
    print("• VAE typically has better diversity (higher IS) but may have")
    print("  higher reconstruction error compared to AE")
    
    print("\n4. FILES GENERATED:")
    print("-" * 40)
    for root, dirs, files in os.walk('outputs'):
        for file in files:
            if file.endswith(('.png', '.json')):
                print(f"  • {os.path.join(root, file)}")

if __name__ == '__main__':
    display_results()