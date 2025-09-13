import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def show_sample_images():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Create a grid of sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for class_id in range(10):
        # Find the first image of this class
        for i, (image, label) in enumerate(dataset):
            if label == class_id:
                axes[class_id].imshow(image.squeeze(), cmap='gray')
                axes[class_id].set_title(f'Class {class_id}: {class_names[class_id]}')
                axes[class_id].axis('off')
                break
    
    plt.tight_layout()
    plt.savefig('./outputs/figures/sample_images.png')
    plt.close()


def analyze_dataset():
    # Load the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    # Get class distribution
    train_labels = [label for _, label in train_dataset]
    test_labels = [label for _, label in test_dataset]
    
    # Class names for FashionMNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Calculate class distribution
    train_counts = np.bincount(train_labels)
    test_counts = np.bincount(test_labels)
    
    # Create visualizations
    plt.figure(figsize=(12, 5))
    
    # Training set distribution
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(10), train_counts)
    plt.title('Training Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(10), range(10))
    # Add value labels on bars
    for bar, count in zip(bars, train_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(count), ha='center', va='bottom')
    
    # Test set distribution
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(10), test_counts)
    plt.title('Test Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(10), range(10))
    # Add value labels on bars
    for bar, count in zip(bars, test_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./outputs/figures/class_distribution.png')
    plt.close()

    print("=== DATASET ANALYSIS ===")
    print(f"Dataset: FashionMNIST")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Number of classes: {len(class_names)}")
    print("\nClass names:", class_names)
    
    print("\nTraining set class distribution:")
    for i, count in enumerate(train_counts):
        print(f"  Class {i} ({class_names[i]}): {count} samples")
    
    print("\nTest set class distribution:")
    for i, count in enumerate(test_counts):
        print(f"  Class {i} ({class_names[i]}): {count} samples")
    
    # Save statistics to file
    stats = {
        'dataset': 'FashionMNIST',
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'image_shape': list(train_dataset[0][0].shape),
        'num_classes': len(class_names),
        'class_names': class_names,
        'train_distribution': train_counts.tolist(),
        'test_distribution': test_counts.tolist()
    }
    
    import json
    with open('./outputs/metrics/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == '__main__':
    analyze_dataset()