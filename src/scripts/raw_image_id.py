import argparse
import numpy as np
import skdim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate Intrinsic Dimension of raw image datasets.")
    parser.add_argument(
        "-d", 
        "--dataset", 
        type=str, 
        choices=["MNIST", "FMNIST", "CIFAR10"],
        help="The name of the dataset to evaluate", 
        required=True
    )
    return parser.parse_args()

def load_dataset(dataset_name):
    # Standard transforms: convert to tensor and flatten
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    if dataset_name == "MNIST":
        ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "FMNIST":
        ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return ds

def main():
    args = parse_args()
    print(f"Loading raw {args.dataset} dataset...")
    
    dataset = load_dataset(args.dataset)
    
    # Use a dataloader to efficiently grab a large batch
    # Using 10,000 samples is generally sufficient for a stable ID estimate
    # while keeping computation time reasonable.
    loader = DataLoader(dataset, batch_size=10000, shuffle=True)
    images, _ = next(iter(loader))
    
    # Convert to numpy array for skdim
    image_matrix = images.numpy()
    
    ambient_dim = image_matrix.shape[1]
    print(f"Sample size: {image_matrix.shape[0]}")
    print(f"Ambient Dimension (Raw Pixels): {ambient_dim}")
    print("\nCalculating Intrinsic Dimension (ID)...")

    # 1. Maximum Likelihood Estimator (Levina-Bickel)
    try:
        mle_id = skdim.id.MLE().fit_transform(image_matrix)
        print(f"Levina-Bickel MLE Estimated ID: {mle_id:.4f}")
    except Exception as e:
        print(f"MLE estimation failed: {e}")

    # 2. Two-NN Estimator
    try:
        twonn_id = skdim.id.TwoNN().fit_transform(image_matrix)
        print(f"Two-NN Estimated ID:          {twonn_id:.4f}")
    except Exception as e:
        print(f"Two-NN estimation failed: {e}")

if __name__ == "__main__":
    main()