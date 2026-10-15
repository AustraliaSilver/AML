"""Guards for the multi-dataset task registry and the shared-head class ranges.

Written as unittest.TestCase (not bare pytest functions) because CI runs
`python -m unittest discover tests`, which does not collect pytest-style
functions. A guard that CI never executes is not a guard.
"""
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from PIL import Image

from mgn.blocks import ConvAML_18Layer_Hybrid
from mgn.datasets import (
    TASK_SPECS,
    TinyImageNet,
    class_counts,
    num_classes_for,
    resolve_task_root,
    task_spec,
)
from train_continual_aml import _ROOT, build_arg_parser, class_range_ce, evaluate


def _write_image(path, size=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (size, size), color=(10, 20, 30)).save(path)


def _make_tinyimagenet(root, val_layout="annotations"):
    """Minimal but faithful Tiny ImageNet directory release."""
    base = os.path.join(root, "tiny-imagenet-200")
    _write_image(os.path.join(base, "train", "n00000002", "images", "b0.JPEG"))
    _write_image(os.path.join(base, "train", "n00000002", "images", "b1.JPEG"))
    _write_image(os.path.join(base, "train", "n00000001", "images", "a0.JPEG"))
    if val_layout == "annotations":
        _write_image(os.path.join(base, "val", "images", "v0.JPEG"))
        _write_image(os.path.join(base, "val", "images", "v1.JPEG"))
        with open(os.path.join(base, "val", "val_annotations.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write("v0.JPEG\tn00000001\t0\t0\t64\t64\n")
            fh.write("v1.JPEG\tn00000002\t1\t0\t64\t64\n")
    else:
        _write_image(os.path.join(base, "val", "n00000001", "images", "v0.JPEG"))
        _write_image(os.path.join(base, "val", "n00000002", "images", "v1.JPEG"))
    return base


def _make_tinyimagenet_numeric(root):
    """Mirrors the repo's data/tiny-imagenet export: numeric class folders,
    PNGs directly under them, and `valid/` rather than `val/`."""
    base = os.path.join(root, "tiny-imagenet")
    for cls in ("0", "2", "10"):
        _write_image(os.path.join(base, "train", cls, "a.png"))
        _write_image(os.path.join(base, "valid", cls, "b.png"))
    return base


class TestTaskRegistry(unittest.TestCase):
    def test_expected_tasks_and_counts(self):
        self.assertEqual(
            class_counts(["cifar10", "cifar100", "tinyimagenet"]), [10, 100, 200])
        self.assertEqual(
            num_classes_for(["cifar10", "cifar100", "tinyimagenet"]), 200)
        self.assertEqual(TASK_SPECS["tinyimagenet"].input_size, 64)
        self.assertEqual(TASK_SPECS["cifar100"].input_size, 32)

    def test_default_sequence_is_still_ten_classes(self):
        # The paper's protocol. A drift here would silently change every number.
        self.assertEqual(num_classes_for(["mnist", "cifar10", "fashion"]), 10)

    def test_unknown_task_names_the_known_ones(self):
        with self.assertRaises(ValueError) as cm:
            task_spec("cifar1000")
        self.assertIn("cifar100", str(cm.exception))

    def test_cli_defaults_are_unchanged(self):
        # --tasks and --data-root are what every README number was measured
        # with; a drifting default silently changes which method was measured.
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.tasks, "mnist,cifar10,fashion")
        self.assertEqual(args.data_root, os.path.join(_ROOT, "data"))


class TestClassRangeCE(unittest.TestCase):
    def test_full_width_single_count_is_bit_identical(self):
        torch.manual_seed(0)
        lg = torch.randn(8, 10)
        y = torch.randint(0, 10, (8,))
        got = class_range_ce(lg, y, torch.full((8,), 10))
        want = F.cross_entropy(lg.float(), y, label_smoothing=0.1)
        self.assertTrue(torch.equal(got, want))

    def test_columns_outside_the_task_range_do_not_affect_the_loss(self):
        torch.manual_seed(0)
        lg = torch.randn(6, 20)
        y = torch.randint(0, 10, (6,))
        base = class_range_ce(lg, y, torch.full((6,), 10))
        tampered = lg.clone()
        tampered[:, 10:] = 1e4        # would dominate if those columns counted
        got = class_range_ce(tampered, y, torch.full((6,), 10))
        self.assertTrue(torch.allclose(base, got))

    def test_mixed_counts_score_each_sample_over_its_own_range(self):
        lg = torch.zeros(4, 200)
        lg[:, 0] = 1.0                # class 0 is the argmax of every prefix
        y = torch.zeros(4, dtype=torch.long)
        mixed = class_range_ce(lg, y, torch.tensor([10, 10, 100, 200]))
        # Scoring the 10-class rows over all 200 columns is strictly worse:
        # the label-smoothing mass is spread over 190 classes the task has
        # never seen.
        wrong = class_range_ce(lg, y, torch.full((4,), 200))
        self.assertLess(mixed.item(), wrong.item())


class TestTinyImageNetLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_split_and_sorted_labels(self):
        base = _make_tinyimagenet(self.tmp.name)
        ds = TinyImageNet(base, train=True)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds.classes, ["n00000001", "n00000002"])
        self.assertEqual(sorted(lbl for _, lbl in ds.samples), [0, 1, 1])

    def test_locate_from_a_parent_directory(self):
        _make_tinyimagenet(self.tmp.name)
        # A Kaggle mount hands you the parent (…/input/<dataset>), not the
        # release folder itself.
        ds = TinyImageNet(self.tmp.name, train=True)
        self.assertEqual(len(ds), 3)

    def test_val_from_annotations(self):
        base = _make_tinyimagenet(self.tmp.name)
        ds = TinyImageNet(base, train=False)
        self.assertEqual(len(ds), 2)
        self.assertEqual(sorted(lbl for _, lbl in ds.samples), [0, 1])

    def test_val_from_class_folders(self):
        base = _make_tinyimagenet(self.tmp.name, val_layout="folders")
        ds = TinyImageNet(base, train=False)
        self.assertEqual(len(ds), 2)

    def test_getitem_returns_a_transformed_tensor(self):
        import torchvision.transforms as T
        base = _make_tinyimagenet(self.tmp.name)
        ds = TinyImageNet(base, train=True, transform=T.ToTensor())
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (3, 64, 64))
        self.assertIsInstance(y, int)

    def test_get_task_data_serves_the_new_branch(self):
        from train_continual_aml import get_task_data
        base = _make_tinyimagenet(self.tmp.name)
        _tl, el = get_task_data("tinyimagenet", base, workers=0)
        x, y = next(iter(el))
        self.assertEqual(tuple(x.shape[1:]), (3, 64, 64))
        self.assertTrue(y.max().item() <= 1)

    def test_valid_dir_and_numeric_class_order(self):
        # The export actually present in this repo's data/tiny-imagenet: numeric
        # class folders, PNGs directly under them, and `valid/` not `val/`.
        base = _make_tinyimagenet_numeric(self.tmp.name)
        tr = TinyImageNet(base, train=True)
        va = TinyImageNet(base, train=False)
        self.assertEqual(tr.classes, ["0", "2", "10"])   # numeric, not lexicographic
        self.assertEqual(tr.class_to_idx, {"0": 0, "2": 1, "10": 2})
        self.assertEqual(len(tr), 3)
        self.assertEqual(len(va), 3)
        self.assertEqual(va.class_to_idx, tr.class_to_idx)


class TestMultiRootResolution(unittest.TestCase):
    """`--data-root` may be a pathsep list; each task uses the root holding it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_root_is_returned_verbatim(self):
        s = os.path.join(self.tmp.name, "only")
        self.assertEqual(resolve_task_root("cifar10", s), s)

    def test_picks_the_root_that_holds_the_task(self):
        empty = os.path.join(self.tmp.name, "empty")
        good = os.path.join(self.tmp.name, "good")
        os.makedirs(os.path.join(good, "cifar-10-batches-py"))
        os.makedirs(empty)
        self.assertEqual(
            resolve_task_root("cifar10", os.pathsep.join([empty, good])), good)

    def test_falls_back_to_the_first_root(self):
        a = os.path.join(self.tmp.name, "a")
        b = os.path.join(self.tmp.name, "b")
        os.makedirs(a)
        os.makedirs(b)
        self.assertEqual(resolve_task_root("cifar10", os.pathsep.join([a, b])), a)

    def test_tiny_imagenet_is_found_under_a_parent_mount(self):
        base = _make_tinyimagenet(self.tmp.name)
        bogus = os.path.join(self.tmp.name, "bogus")
        os.makedirs(bogus)
        joined = os.pathsep.join([bogus, os.path.dirname(base)])
        self.assertEqual(
            resolve_task_root("tinyimagenet", joined), os.path.dirname(base))


class TestSharedHeadClassRanges(unittest.TestCase):
    """One head of width max(n_classes); each task read back on its prefix."""

    @staticmethod
    def _loader(n_classes=10, n=8, size=32):
        x = torch.randn(n, 3, size, size)
        y = torch.randint(0, n_classes, (n,))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=n)

    def test_head_can_be_200_wide_at_32_and_64(self):
        m = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=16, num_classes=200)
        m.eval()
        for size in (32, 64):
            with torch.no_grad():
                out, _, _, _ = m(torch.randn(2, 3, size, size), tau=0.1)
            self.assertEqual(tuple(out.shape), (2, 200))

    def test_evaluate_reads_back_only_the_task_range(self):
        torch.manual_seed(0)
        m = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=16, num_classes=200)
        loader = self._loader(n_classes=10)
        # Push every column >= 10 far above the rest, so a whole-head argmax
        # can never land in the task's range. n_classes must therefore be
        # honoured for the accuracy to be anything but zero.
        with torch.no_grad():
            m.proto_weights[:, m.wo_size + 10: m.wo_size + 200] = 1e4

        acc_prefix, _ = evaluate(m, loader, torch.device("cpu"), n_classes=10)
        acc_full, _ = evaluate(m, loader, torch.device("cpu"))
        self.assertEqual(acc_full, 0.0)

        # The prefix accuracy must equal an explicit [:, :10].argmax reference.
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                out, _, _, _ = m(x, tau=0.1)
                correct += (out[:, :10].argmax(-1) == y).sum().item()
                total += y.numel()
        self.assertEqual(acc_prefix, correct / total)

    def test_none_matches_the_old_behaviour_on_a_10_class_head(self):
        torch.manual_seed(0)
        m = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=16, num_classes=10)
        loader = self._loader(n_classes=10)
        a, _ = evaluate(m, loader, torch.device("cpu"))
        b, _ = evaluate(m, loader, torch.device("cpu"), n_classes=10)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
