import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_continual_dataloaders(data_dir='./data', batch_size=128, test_batch_size=256, num_workers=2):
    """
    Constructs optimized DataLoaders for MNIST, CIFAR-10, and Fashion-MNIST.
    """
    tf_mnist_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tf_mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tf_cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tf_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    tf_fashion_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    tf_fashion_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    loaders = {
        "mnist": {
            "train": DataLoader(datasets.MNIST(data_dir, True, download=True, transform=tf_mnist_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            "test": DataLoader(datasets.MNIST(data_dir, False, download=True, transform=tf_mnist_test), batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        },
        "cifar": {
            "train": DataLoader(datasets.CIFAR10(data_dir, True, download=True, transform=tf_cifar_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            "test": DataLoader(datasets.CIFAR10(data_dir, False, download=True, transform=tf_cifar_test), batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        },
        "fashion": {
            "train": DataLoader(datasets.FashionMNIST(data_dir, True, download=True, transform=tf_fashion_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            "test": DataLoader(datasets.FashionMNIST(data_dir, False, download=True, transform=tf_fashion_test), batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        }
    }
    return loaders
