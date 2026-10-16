"""Contracts of the data pipeline that the published tables rest on.

Each of these used to exist in two places, and each pair could drift with no test
noticing:

  * the per-task normalisation constants lived in `mgn/datasets.TASK_SPECS` AND
    as literals inside `train_continual_aml.get_task_data` (10 numbers across 4
    branches). `--input-norm` reads the spec while the loaders applied the
    literals, so the "one task-agnostic rule" run and the per-task run could have
    disagreed about what the *source* normalisation even was.
  * the task-name -> torchvision-class map existed in `main()` and, inverted, in
    `build_replay_buffer()`.
  * the replay path rebuilt two lookup tables inside every training step, paying
    one device->host sync (`src.unique().tolist()`) and one sync PER EXEMPLAR
    (`int(t_)` over ~48 replayed task ids).

The tests compare against the historical constants, so deliberately changing
TASK_SPECS (which would move every published number) has to be acknowledged here.
"""

import os
import unittest

import torch
import torchvision


def _trainer_src():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "train_continual_aml.py"), encoding="utf-8") as f:
        return f.read()


# Historical per-task normalisation, exactly as it was hardcoded in the trainer.
HISTORICAL = {
    "mnist": ((0.1307,) * 3, (0.3081,) * 3),
    "fashion": ((0.2860,) * 3, (0.3530,) * 3),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
}

NORM_LITERALS = [f"{v}" for m, s in HISTORICAL.values() for v in m + s]


class TestSpecIsTheOnlySource(unittest.TestCase):

    def test_spec_still_holds_the_published_constants(self):
        from mgn.datasets import task_spec
        for task, (mean, std) in HISTORICAL.items():
            spec = task_spec(task)
            self.assertEqual(tuple(spec.mean), mean, f"{task} mean moved")
            self.assertEqual(tuple(spec.std), std, f"{task} std moved")

    def test_trainer_hardcodes_no_normalisation_literal(self):
        """The regression guard for the refactor: if a constant reappears in the
        trainer it is a second source of truth again, and the two can disagree
        without any test failing."""
        src = _trainer_src()
        for lit in sorted(set(NORM_LITERALS)):
            self.assertNotIn(
                lit, src,
                f"train_continual_aml.py hardcodes the normalisation constant "
                f"{lit}; read it from mgn.datasets.TASK_SPECS instead")

    def test_trainer_derives_mean_std_from_the_spec(self):
        src = _trainer_src()
        self.assertIn("mean, std = _spec.mean, _spec.std", src)

    def test_dataset_class_map_is_consistent(self):
        from train_continual_aml import _DATASET_CLASS, _SPEC_OF_DATASET_CLASS
        from mgn.datasets import TASK_SPECS
        for task in _DATASET_CLASS:
            self.assertIn(task, TASK_SPECS)
        for cls in _DATASET_CLASS.values():
            self.assertTrue(hasattr(torchvision.datasets, cls),
                            f"{cls} is not a torchvision dataset class")
        self.assertEqual(
            _SPEC_OF_DATASET_CLASS,
            {cls: t for t, cls in _DATASET_CLASS.items()})
        # Tiny ImageNet is served by directory release, so it is the one task
        # without a torchvision class -- and the one replay refuses.
        self.assertEqual(set(_DATASET_CLASS), set(TASK_SPECS) - {"tinyimagenet"})

    def test_main_and_replay_share_the_map(self):
        """`main()` may not carry its own copy of the mapping: the literal lives
        in exactly one place, and `main()` resolves through it."""
        src = _trainer_src()
        self.assertIn("_DATASET_CLASS.get(name)", src)
        self.assertEqual(src.count('"fashion": "FashionMNIST"'), 1)


class MNIST:
    """Only what `build_replay_buffer` touches: the class NAME and .data/.targets.

    The name must be exactly the torchvision one -- `build_replay_buffer` keys the
    pixel layout off `type(dataset).__name__` -- so these stand-ins are named
    MNIST / CIFAR10 / TinyImageNet rather than `_Fake*`.
    """

    def __init__(self, n=64):
        g = torch.Generator().manual_seed(1)
        self.data = (torch.rand(n, 28, 28, generator=g) * 255).to(torch.uint8)
        self.targets = torch.arange(n) % 10


class CIFAR10:
    def __init__(self, n=64):
        g = torch.Generator().manual_seed(2)
        self.data = ((torch.rand(n, 32, 32, 3, generator=g) * 255)
                     .to(torch.uint8).numpy())
        self.targets = torch.arange(n) % 10


class TinyImageNet:
    def __init__(self):
        self.data = None
        self.targets = torch.arange(4)


class TestReplayUsesTheSameRule(unittest.TestCase):
    """An exemplar and a freshly loaded image of the same task must be prepared
    by ONE rule -- that identity is what `check_input_norm_args` protects."""

    def _old_selection(self, ds, per_class, num_classes):
        """The index selection, reproduced: randperm off the global RNG."""
        targets = torch.as_tensor(ds.targets)
        chosen = []
        for c in range(num_classes):
            cls_idx = (targets == c).nonzero(as_tuple=True)[0]
            chosen.append(cls_idx[torch.randperm(len(cls_idx))[:per_class]])
        return torch.cat(chosen)

    def test_mnist_exemplars_equal_the_spec_rule(self):
        from train_continual_aml import build_replay_buffer
        ds = MNIST()
        torch.manual_seed(1234)
        xs, ys = build_replay_buffer(ds, per_class=4, num_classes=10)
        torch.manual_seed(1234)
        idx = self._old_selection(ds, 4, 10)
        raw = ds.data[idx].float().unsqueeze(1) / 255.0
        data_t = torch.nn.functional.interpolate(
            raw, size=(32, 32), mode="bilinear").repeat(1, 3, 1, 1)
        self.assertEqual(xs.shape, (40, 3, 32, 32))
        self.assertTrue(torch.equal((data_t - 0.1307) / 0.3081, xs))
        self.assertTrue(torch.equal(torch.as_tensor(ds.targets)[idx], ys))

    def test_cifar_exemplars_equal_the_spec_rule(self):
        from train_continual_aml import build_replay_buffer
        ds = CIFAR10()
        torch.manual_seed(1234)
        xs, _ys = build_replay_buffer(ds, per_class=4, num_classes=10)
        torch.manual_seed(1234)
        idx = self._old_selection(ds, 4, 10)
        data_t = torch.from_numpy(ds.data[idx.numpy()]).permute(
            0, 3, 1, 2).float() / 255.0
        mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        std = torch.tensor((0.2023, 0.1994, 0.2010)).view(1, 3, 1, 1)
        self.assertTrue(torch.equal((data_t - mean) / std, xs))

    def test_unsupported_dataset_still_names_the_flag(self):
        from train_continual_aml import build_replay_buffer
        with self.assertRaises(ValueError) as ctx:
            build_replay_buffer(TinyImageNet(), per_class=1, num_classes=4)
        self.assertIn("Tiny ImageNet", str(ctx.exception))
        self.assertIn("--replay-per-class 0", str(ctx.exception))


class TestReplayHotPathHasNoPerStepSync(unittest.TestCase):
    """Source guard, in the style of the other source-inspection tests: the two
    per-step syncs must not come back into `train_task`'s loop."""

    def test_no_unique_tolist_and_no_per_exemplar_int(self):
        src = _trainer_src()
        self.assertNotIn("src.unique().tolist()", src)
        self.assertNotIn("for t_ in re_t", src)
        self.assertNotIn("task_class_counts[int(t_)]", src)

    def test_lookup_tables_are_built_before_the_epoch_loop(self):
        """Both tables must be built once per task, not once per step. The second
        occurrence of the pattern was inside `consolidate()`, and this guard is
        what found it."""
        src = _trainer_src()
        train = src[src.index("def train_task("):]
        loop = train.index("for epoch in range(epochs):")
        self.assertLess(train.index("row_task = torch.full"), loop)
        self.assertLess(train.index("counts_tab = torch.tensor"), loop)
        cons = src[src.index("def consolidate("):src.index("def train_task(")]
        self.assertLess(cons.index("counts_tab = torch.tensor"),
                        cons.index("for _ in range(epochs):"))


class TestEpochListValidation(unittest.TestCase):
    """`--epochs` was a bare `assert`, which `python -O` removes and which gave an
    IndexError far from the cause when the list was too short."""

    def test_valid_schedules_pass_through(self):
        from train_continual_aml import resolve_epoch_list
        tasks = ["mnist", "cifar10", "fashion"]
        self.assertEqual(resolve_epoch_list("5,28,16", tasks), [5, 28, 16])
        self.assertEqual(resolve_epoch_list(" 5, 28 ,16 ", tasks), [5, 28, 16])
        self.assertEqual(resolve_epoch_list("3", ["mnist"]), [3])

    def test_too_few_epochs_is_refused_with_both_counts(self):
        from train_continual_aml import resolve_epoch_list
        tasks = ["mnist", "cifar10", "fashion"]
        with self.assertRaises(SystemExit) as ctx:
            resolve_epoch_list("5,28", tasks)
        msg = str(ctx.exception)
        self.assertIn("has 2 value(s)", msg)
        self.assertIn("3 task(s)", msg)
        self.assertIn("--epochs 5,28,16", msg)

    def test_too_many_epochs_is_refused_not_ignored(self):
        from train_continual_aml import resolve_epoch_list
        with self.assertRaises(SystemExit) as ctx:
            resolve_epoch_list("5,28,16,99", ["mnist", "cifar10", "fashion"])
        self.assertIn("has 4 value(s)", str(ctx.exception))

    def test_non_integer_is_refused(self):
        from train_continual_aml import resolve_epoch_list
        with self.assertRaises(SystemExit) as ctx:
            resolve_epoch_list("5,twenty", ["mnist", "cifar10"])
        self.assertIn("comma-separated list of integers", str(ctx.exception))

    def test_zero_epochs_is_refused(self):
        """0 epochs would leave the task untrained while still printing a row of
        the accuracy matrix for it."""
        from train_continual_aml import resolve_epoch_list
        with self.assertRaises(SystemExit) as ctx:
            resolve_epoch_list("5,0,16", ["mnist", "cifar10", "fashion"])
        self.assertIn("positive", str(ctx.exception))

    def test_main_uses_the_validated_helper(self):
        self.assertIn("ep_list = resolve_epoch_list(args.epochs, tasks)",
                      _trainer_src())
        self.assertNotIn("assert len(ep_list) == len(tasks)", _trainer_src())


if __name__ == "__main__":
    unittest.main()
