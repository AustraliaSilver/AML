"""Per-task dataset registry for the continual-learning trainer.

ONE source of truth for two things that used to be hardcoded in the trainer and
therefore could drift:

  * how many classes a task has, and
  * the native input resolution it is trained at.

The trainer builds ONE supernet for the whole sequence and predicts task
identity at test time (the Domain-IL / Class-IL protocol), so the classifier
head has a single width: ``max`` over the tasks of their class count. A task
only ever supervises and reads back a PREFIX of that width -- its own class
range ``[0, n_classes)``. Prototype rows are isolated per task
(``ProtoIsolator``), so two tasks that both use slot 0 never interfere: they
read different rows. This is what lets CIFAR-10 (10 classes), CIFAR-100 (100)
and Tiny ImageNet (200) share one model without a per-task head width, which
would change the flat prototype layout and every CUDA offset with it.

The resolution is per task because the architecture does not depend on it: the
conv trunk is stride-1 with fixed pooling at layers 5 and 11, the classifier
head is ``AdaptiveAvgPool2d((4, 4))`` and the vision encoder is
``AdaptiveAvgPool2d((2, 2))``, so a 32x32 and a 64x64 task can share one trunk
and one router. Keeping the native size avoids throwing away Tiny ImageNet's
detail by downsampling it to 32x32.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from torch.utils.data import Dataset

# Per-channel statistics. CIFAR-10's live in the trainer (they predate this
# module and are quoted by the paper); the rest are the standard values for
# their dataset.
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)


@dataclass(frozen=True)
class TaskSpec:
    """Class count, native input side and normalisation for one task."""

    n_classes: int
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


TASK_SPECS: Dict[str, TaskSpec] = {
    "mnist": TaskSpec(10, 32, (0.1307,) * 3, (0.3081,) * 3),
    "fashion": TaskSpec(10, 32, (0.2860,) * 3, (0.3530,) * 3),
    "cifar10": TaskSpec(10, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": TaskSpec(100, 32, CIFAR100_MEAN, CIFAR100_STD),
    "tinyimagenet": TaskSpec(200, 64, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
}


def task_spec(name: str) -> TaskSpec:
    """Return the spec for a task, with an actionable error for a typo."""
    try:
        return TASK_SPECS[name]
    except KeyError:
        known = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"unknown task {name!r}; known tasks: {known}") from None


def class_counts(tasks: List[str]) -> List[int]:
    """Per-task class counts, in sequence order."""
    return [task_spec(t).n_classes for t in tasks]


def num_classes_for(tasks: List[str]) -> int:
    """Width of the single shared classifier head for this sequence."""
    return max(class_counts(tasks))


def dataset_present(name: str, root: str) -> bool:
    """True when `root` already holds this task's data (download is a no-op)."""
    if name == "mnist":
        return os.path.isdir(os.path.join(root, "MNIST"))
    if name == "fashion":
        return os.path.isdir(os.path.join(root, "FashionMNIST"))
    if name == "cifar10":
        return os.path.isdir(os.path.join(root, "cifar-10-batches-py"))
    if name == "cifar100":
        return os.path.isdir(os.path.join(root, "cifar-100-python"))
    if name == "tinyimagenet":
        try:
            TinyImageNet._locate(root)
            return True
        except FileNotFoundError:
            return False
    return False


def resolve_task_root(name: str, data_root: str) -> str:
    """Pick the root that holds `name` from an os.pathsep-separated list.

    A single root (the default) is returned unchanged, so existing runs are
    bit-for-bit unaffected. Several are useful when the datasets live in
    different places: on Kaggle each dataset is mounted under its own
    ``/kaggle/input/<slug>``, so one ``--data-root`` listing all the mounts
    finds each task where it actually is. When none of the roots already holds
    the data the first one is returned, so ``download=True`` can fetch it.
    """
    roots = [r for r in data_root.split(os.pathsep) if r]
    if len(roots) <= 1:
        return data_root
    for r in roots:
        if dataset_present(name, r):
            return r
    return roots[0]


_IMG_SUFFIXES = (".jpeg", ".jpg", ".png")


def _list_images(directory: str) -> List[str]:
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(_IMG_SUFFIXES)
    )


def _sorted_wnids(names) -> List[str]:
    """Sorted class-folder names, numeric ones numerically.

    The Stanford release uses wnids (``n01443537``), which sort fine as text.
    Some preprocessed exports name the folders ``0``..``199`` instead, and
    sorting those lexicographically would scatter the label order ("10" before
    "2") for no reason. Train and val share whichever order this returns, so a
    label means the same class in both.
    """
    names = list(names)
    if names and all(n.isdigit() for n in names):
        return sorted(names, key=int)
    return sorted(names)


class TinyImageNet(Dataset):
    """Tiny ImageNet (200 classes, 64x64) from the usual directory release.

    torchvision ships no Tiny ImageNet, so this reads the on-disk layout that the
    Stanford tarball, the common Kaggle mirrors and the preprocessed exports all
    use::

        <root>/train/<wnid>[/images]/*.JPEG
        <root>/val|valid/<wnid>[/images]/*.JPEG          (per-class folders)
        <root>/val/images/*.JPEG + val_annotations.txt   (Stanford release)

    The validation folder is accepted as either ``val`` or ``valid``, the
    per-class ``images`` sub-step is optional, and class folders named with plain
    integers are ordered numerically. ``root`` may be the dataset directory
    itself or any directory that contains it (``/kaggle/input/<dataset>``, say):
    the release folder is found by looking for a ``train/`` child. The class
    order is shared by both splits, so a label means the same thing in train and
    val.
    """

    def __init__(self, root: str, train: bool = True, transform=None):
        self.transform = transform
        base = self._locate(root)
        self.root = base
        train_dir = os.path.join(base, "train")
        wnids = _sorted_wnids(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
        if not wnids:
            raise FileNotFoundError(f"no class folders under {train_dir!r}")
        self.classes = wnids
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}

        self.samples: List[Tuple[str, int]] = []
        if train:
            if not self._collect_class_folders(train_dir, wnids):
                raise FileNotFoundError(
                    f"no images under any class folder in {train_dir!r}")
        else:
            val_dir = next(
                (os.path.join(base, c) for c in ("val", "valid")
                 if os.path.isdir(os.path.join(base, c))), None)
            if val_dir is None:
                raise FileNotFoundError(
                    f"neither val/ nor valid/ exists under {base!r}")
            if not self._collect_class_folders(val_dir, wnids):
                # Stanford release: flat val/images + val_annotations.txt.
                ann = os.path.join(val_dir, "val_annotations.txt")
                img_dir = os.path.join(val_dir, "images")
                if not os.path.isfile(ann):
                    raise FileNotFoundError(
                        f"no per-class folders under {val_dir!r} and no "
                        f"{ann!r}; the val split is not laid out as expected")
                with open(ann, encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 2 or parts[1] not in self.class_to_idx:
                            continue
                        self.samples.append(
                            (os.path.join(img_dir, parts[0]),
                             self.class_to_idx[parts[1]])
                        )

        if not self.samples:
            raise FileNotFoundError(
                f"Tiny ImageNet {'train' if train else 'val'} split under "
                f"{base!r} contains no images")
        self.targets = [lbl for _, lbl in self.samples]

    def _collect_class_folders(self, root_dir: str, wnids: List[str]) -> bool:
        """Collect ``<root_dir>/<wnid>[/images]/*``. True if any image found."""
        found = False
        for w in wnids:
            img_dir = os.path.join(root_dir, w, "images")
            if not os.path.isdir(img_dir):
                img_dir = os.path.join(root_dir, w)
            if not os.path.isdir(img_dir):
                continue
            for p in _list_images(img_dir):
                self.samples.append((p, self.class_to_idx[w]))
                found = True
        return found

    @staticmethod
    def _locate(root: str) -> str:
        if os.path.isdir(os.path.join(root, "train")):
            return root
        if os.path.isdir(root):
            for entry in sorted(os.listdir(root)):
                cand = os.path.join(root, entry)
                if os.path.isdir(os.path.join(cand, "train")):
                    return cand
        raise FileNotFoundError(
            f"no Tiny ImageNet release (a directory containing train/) under "
            f"{root!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        # Imported lazily: pillow arrives with torchvision, and importing it at
        # module load would make `import mgn.datasets` pull it in everywhere.
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
