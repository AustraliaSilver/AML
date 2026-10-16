"""`--input-norm`: the task-agnostic normalisation protocol (open item 23 / B71).

Every published number is produced by loaders that normalise each task with its
own mean/std from `mgn.datasets.TASK_SPECS`. That is a privilege the autonomous
protocol does not have, and `A_auto` is measured under it anyway. These tests
cover the machinery that lets the claim be *measured* instead of argued:

  * the pooled constant is a real pooled second moment, not a mean of means;
  * the wrapper round-trips (a rule equal to the source is the identity) and
    applies each rule as advertised;
  * the default leaves the loader untouched -- this is the release gate, because
    every published number depends on it;
  * the replay combination is refused rather than silently mixed.
"""

import os
import unittest

import torch


class TestPooledStats(unittest.TestCase):

    def setUp(self):
        from mgn.datasets import pooled_input_stats, task_spec
        self.pooled = pooled_input_stats
        self.spec = task_spec

    def test_single_task_is_that_task_exactly(self):
        for t in ("mnist", "fashion", "cifar10", "cifar100", "tinyimagenet"):
            mean, std = self.pooled([t])
            s = self.spec(t)
            self.assertEqual(tuple(round(v, 12) for v in mean),
                             tuple(round(v, 12) for v in s.mean))
            self.assertEqual(tuple(round(v, 12) for v in std),
                             tuple(round(v, 12) for v in s.std))

    def test_mean_is_size_weighted_and_within_range(self):
        tasks = ["mnist", "cifar10", "fashion"]
        mean, _std = self.pooled(tasks)
        means = [self.spec(t).mean for t in tasks]
        for c in range(3):
            lo = min(m[c] for m in means)
            hi = max(m[c] for m in means)
            self.assertTrue(lo <= mean[c] <= hi)

    def test_pooling_two_different_means_raises_the_spread(self):
        """The property that makes this a real experiment: no single constant can
        match two domains, so the pooled std exceeds both of theirs. If a "pooled"
        implementation only averaged the means, every image would be pushed off
        its own distribution by the same amount and the test would not catch it,
        but the *second moment* would be wrong."""
        mean, std = self.pooled(["mnist", "cifar10"])
        self.assertGreater(std[0], self.spec("mnist").std[0])
        self.assertGreater(std[0], self.spec("cifar10").std[0])
        # and the exact identity, computed independently here
        n_m, n_c = 60000.0, 50000.0
        tot = n_m + n_c
        for c in range(3):
            m_m, s_m = self.spec("mnist").mean[c], self.spec("mnist").std[c]
            m_c, s_c = self.spec("cifar10").mean[c], self.spec("cifar10").std[c]
            exp_mean = (n_m * m_m + n_c * m_c) / tot
            exp_second = (n_m * (s_m ** 2 + m_m ** 2) + n_c * (s_c ** 2 + m_c ** 2)) / tot
            self.assertAlmostEqual(mean[c], exp_mean, places=10)
            self.assertAlmostEqual(std[c] ** 2, exp_second - exp_mean ** 2, places=10)

    def test_unknown_task_is_named(self):
        with self.assertRaises(ValueError) as ctx:
            self.pooled(["mnist", "nosuchtask"])
        self.assertIn("nosuchtask", str(ctx.exception))


class TestResolveInputNorm(unittest.TestCase):

    def test_per_task_is_none_everywhere(self):
        from train_continual_aml import resolve_input_norm
        self.assertEqual(resolve_input_norm(["mnist", "cifar10"], "per-task"),
                         {"mnist": None, "cifar10": None})

    def test_pooled_is_one_shared_constant_for_the_whole_sequence(self):
        from train_continual_aml import resolve_input_norm
        m = resolve_input_norm(["mnist", "cifar10", "fashion"], "pooled")
        kinds = {v[0] for v in m.values()}
        self.assertEqual(kinds, {"const"})
        self.assertEqual(len({tuple(v[1]) for v in m.values()}), 1,
                         "pooled must be ONE constant, not one per task")

    def test_imagewise_and_none_kinds(self):
        from train_continual_aml import resolve_input_norm
        self.assertEqual(resolve_input_norm(["mnist"], "imagewise")["mnist"][0],
                         "imagewise")
        desc = resolve_input_norm(["mnist"], "none")["mnist"]
        self.assertEqual(desc[0], "const")
        self.assertEqual(desc[1], (0.0, 0.0, 0.0))
        self.assertEqual(desc[2], (1.0, 1.0, 1.0))

    def test_unknown_mode_exits(self):
        from train_continual_aml import resolve_input_norm
        with self.assertRaises(SystemExit):
            resolve_input_norm(["mnist"], "magic")


class TestRenormDataset(unittest.TestCase):

    def setUp(self):
        from train_continual_aml import _RenormDataset
        self.Cls = _RenormDataset
        torch.manual_seed(0)
        self.base = torch.utils.data.TensorDataset(
            torch.randn(5, 3, 8, 8), torch.arange(5))
        self.src_mean, self.src_std = (0.1307,) * 3, (0.3081,) * 3

    def test_same_rule_as_the_source_is_the_identity(self):
        w = self.Cls(self.base, self.src_mean, self.src_std,
                     ("const", self.src_mean, self.src_std))
        self.assertLess((w[2][0] - self.base[2][0]).abs().max().item(), 1e-6)

    def test_none_recovers_the_raw_image(self):
        w = self.Cls(self.base, self.src_mean, self.src_std, ("none", None, None))
        raw = (self.base[1][0] * torch.tensor(self.src_std).view(3, 1, 1)
               + torch.tensor(self.src_mean).view(3, 1, 1))
        self.assertEqual(torch.equal(w[1][0], raw), True)

    def test_const_applies_the_requested_constant(self):
        dst_mean, dst_std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
        w = self.Cls(self.base, self.src_mean, self.src_std,
                     ("const", dst_mean, dst_std))
        raw = (self.base[0][0] * torch.tensor(self.src_std).view(3, 1, 1)
               + torch.tensor(self.src_mean).view(3, 1, 1))
        want = (raw - torch.tensor(dst_mean).view(3, 1, 1)) / torch.tensor(dst_std).view(3, 1, 1)
        self.assertLess((w[0][0] - want).abs().max().item(), 1e-6)

    def test_imagewise_zeroes_each_image(self):
        w = self.Cls(self.base, self.src_mean, self.src_std,
                     ("imagewise", None, None))
        x, _y = w[3]
        self.assertLess(abs(x.mean().item()), 1e-6)
        self.assertAlmostEqual(x.std(unbiased=False).item(), 1.0, places=5)

    def test_len_and_labels_pass_through(self):
        w = self.Cls(self.base, self.src_mean, self.src_std, ("none", None, None))
        self.assertEqual(len(w), 5)
        self.assertEqual([w[i][1].item() for i in range(5)], [0, 1, 2, 3, 4])


class TestLoaderPlumbing(unittest.TestCase):
    """The default must be a no-op: every published number depends on it."""

    def _inject(self, root):
        import train_continual_aml as tca
        base = torch.utils.data.TensorDataset(
            torch.randn(8, 3, 32, 32), torch.zeros(8, dtype=torch.long))
        tca._DATA_CACHE[("mnist", os.path.abspath(root))] = (base, base)
        return tca, base

    def test_default_returns_the_cached_dataset_unwrapped(self):
        root = os.path.join(os.path.abspath(os.sep), "nonexistent-root-for-test")
        tca, base = self._inject(root)
        try:
            tl, el = tca.get_task_data("mnist", root, workers=0, batch_size=4)
            self.assertIs(tl.dataset, base)
            self.assertIs(el.dataset, base)
        finally:
            tca._DATA_CACHE.clear()

    def test_a_task_agnostic_rule_wraps_the_cached_dataset(self):
        root = os.path.join(os.path.abspath(os.sep), "nonexistent-root-for-test")
        tca, base = self._inject(root)
        try:
            tl, el = tca.get_task_data("mnist", root, workers=0, batch_size=4,
                                       inorm=("imagewise", None, None))
            self.assertIsInstance(tl.dataset, tca._RenormDataset)
            self.assertIsInstance(el.dataset, tca._RenormDataset)
            self.assertIs(tl.dataset.base, base)
            x, _y = next(iter(tl))
            self.assertEqual(x.shape, (4, 3, 32, 32))
        finally:
            tca._DATA_CACHE.clear()

    def test_the_cache_keeps_per_task_tensors_only(self):
        """If the cache stored wrapped tensors, the second call with a different
        mode would silently return the first mode's data."""
        root = os.path.join(os.path.abspath(os.sep), "nonexistent-root-for-test")
        tca, base = self._inject(root)
        try:
            tca.get_task_data("mnist", root, workers=0, batch_size=4,
                              inorm=("none", None, None))
            _tr, te = tca._DATA_CACHE[("mnist", os.path.abspath(root))]
            self.assertIs(te, base)
            self.assertNotIsInstance(te, tca._RenormDataset)
        finally:
            tca._DATA_CACHE.clear()


class TestCliContract(unittest.TestCase):

    def test_default_is_per_task(self):
        """The default is the scientific claim: `per-task` is what every README
        number was measured with, and a drifting default would silently turn the
        published tables into a task-agnostic protocol."""
        from train_continual_aml import build_arg_parser
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.input_norm, 'per-task')

    def test_replay_is_refused_rather_than_mixed(self):
        from train_continual_aml import check_input_norm_args
        check_input_norm_args('per-task', use_replay=True)   # allowed
        check_input_norm_args('pooled', use_replay=False)    # allowed
        with self.assertRaises(SystemExit) as ctx:
            check_input_norm_args('pooled', use_replay=True)
        self.assertIn("replay-per-class 0", str(ctx.exception))

    def test_help_mentions_the_flag(self):
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "train_continual_aml.py", "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertIn("--input-norm", r.stdout)
        self.assertIn("A_auto claims to measure", r.stdout)


if __name__ == "__main__":
    unittest.main()
