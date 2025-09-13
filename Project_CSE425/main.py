import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.train import train_vae, train_ae
from src.evaluate import generate_samples_vae, generate_samples_ae
from src.utils import create_dirs

def main():
    create_dirs()
    print("Training VAE...")
    train_vae()
    print("Training AE...")
    train_ae()
    print("Generating samples...")
    generate_samples_vae()
    generate_samples_ae()
    print("Done!")

if __name__ == '__main__':
    main()