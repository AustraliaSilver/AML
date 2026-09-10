import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from collections import Counter
import re, os, pickle

STATS_DIR = os.path.join(".", "data")


def cifar_hflip_flat(x):
    """MetaTrainer(preprocess_fn=...): 50% per-sample horizontal flip on a
    flat [B, 3072] CIFAR batch. Uses the TORCH RNG, so set_seed(seed) keeps
    augmented runs reproducible (the python `random` module used by older
    entry points was left unseeded)."""
    if x.dim() == 2 and x.shape[1] == 3072:
        img = x.reshape(-1, 3, 32, 32)
        flip = torch.rand(img.size(0), device=img.device) < 0.5
        img = torch.where(flip.view(-1, 1, 1, 1), img.flip(-1), img)
        return img.reshape(-1, 3072)
    return x


def _load_or_build_vocab(name, max_features):
    """BOW vocabulary built ONCE from the train split and shared by all
    splits: independent per-split vocabs made column k of the feature matrix
    mean different WORDS in train vs test (silent feature misalignment)."""
    os.makedirs(STATS_DIR, exist_ok=True)
    cache = os.path.join(STATS_DIR, f"bow_vocab_{name}_{max_features}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    from datasets import load_dataset
    hf_train = load_dataset(name, split="train", trust_remote_code=True)
    vocab = Counter()
    for example in hf_train:
        vocab.update(re.findall(r"\w+", example["text"].lower()))
    top_words = [w for w, _ in vocab.most_common(max_features - 1)]
    word2idx = {w: i + 1 for i, w in enumerate(top_words)}
    with open(cache, "wb") as f:
        pickle.dump(word2idx, f)
    return word2idx


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

        # Standardize with stats computed from the TRAIN split for BOTH
        # splits (per-split self-stats leak test data and descale features).
        os.makedirs(STATS_DIR, exist_ok=True)
        stats_path = os.path.join(STATS_DIR, f"flat_stats_{name}.pt")
        if train:
            x_mean = self.data.mean(dim=0, keepdim=True)
            x_std = self.data.std(dim=0, keepdim=True) + 1e-8
            torch.save((x_mean, x_std), stats_path)
        elif os.path.exists(stats_path):
            x_mean, x_std = torch.load(stats_path, weights_only=True)
        else:
            # get_dataset() always builds train first; if out of build order,
            # compute from the paired train split.
            ref = FlattenedImageDataset(name, train=True, max_samples=None, download=download)
            x_mean = ref.data.mean(dim=0, keepdim=True)
            x_std = ref.data.std(dim=0, keepdim=True) + 1e-8
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

        word2idx = _load_or_build_vocab(name, max_features)

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
