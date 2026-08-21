import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from collections import Counter
import re, os, pickle, itertools


class FlattenedImageDataset(Dataset):
    def __init__(self, name, train=True, max_samples=None, download=True):
        self.name = name
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if name in ("mnist", "fashion_mnist") else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if name == "mnist":
            ds = datasets.MNIST(root="./data", train=train, transform=transform, download=download)
        elif name == "fashion_mnist":
            ds = datasets.FashionMNIST(root="./data", train=train, transform=transform, download=download)
        elif name == "cifar10":
            ds = datasets.CIFAR10(root="./data", train=train, transform=transform, download=download)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        self.images = []
        self.labels = []
        for i, (img, lbl) in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            self.images.append(img.flatten().numpy())
            self.labels.append(lbl)

        self.data = torch.tensor(np.array(self.images), dtype=torch.float32)
        self.targets = torch.tensor(self.labels, dtype=torch.long)

        x_mean = self.data.mean(dim=0, keepdim=True)
        x_std = self.data.std(dim=0, keepdim=True) + 1e-8
        self.data = (self.data - x_mean) / x_std

        self.input_dim = self.data.shape[1]
        self.output_dim = len(ds.classes)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TextBOWDataset(Dataset):
    def __init__(self, name, train=True, max_samples=None, max_features=5000, download=True):
        from datasets import load_dataset
        self.name = name
        split = "train" if train else "test" if name == "ag_news" else ("train" if train else "test")

        if name == "ag_news":
            hf_ds = load_dataset("ag_news", split=split, trust_remote_code=True)
        elif name == "imdb":
            hf_ds = load_dataset("imdb", split=split if train else "test", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown text dataset: {name}")

        texts, labels = [], []
        for i, example in enumerate(hf_ds):
            if max_samples and i >= max_samples:
                break
            texts.append(example["text"])
            labels.append(example["label"])

        vocab = Counter()
        for t in texts:
            tokens = re.findall(r"\w+", t.lower())
            vocab.update(tokens)

        top_words = [w for w, _ in vocab.most_common(max_features - 1)]
        word2idx = {w: i + 1 for i, w in enumerate(top_words)}

        X = np.zeros((len(texts), max_features), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = re.findall(r"\w+", t.lower())
            for w in tokens:
                if w in word2idx:
                    X[i, word2idx[w]] += 1.0

        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)
        self.input_dim = max_features
        self.output_dim = np.max(labels) + 1

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_dataset(name, max_train=None, max_test=None, max_features=5000):
    if name in ("mnist", "fashion_mnist", "cifar10"):
        train_ds = FlattenedImageDataset(name, train=True, max_samples=max_train)
        test_ds = FlattenedImageDataset(name, train=False, max_samples=max_test)
    elif name in ("ag_news", "imdb"):
        train_ds = TextBOWDataset(name, train=True, max_samples=max_train, max_features=max_features)
        test_ds = TextBOWDataset(name, train=False, max_samples=max_test, max_features=max_features)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return {
        "name": name,
        "input_dim": train_ds.input_dim,
        "output_dim": train_ds.output_dim,
        "train_x": train_ds.data,
        "train_y": train_ds.targets,
        "test_x": test_ds.data,
        "test_y": test_ds.targets,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
    }
