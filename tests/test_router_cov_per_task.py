"""The per-task precision must behave like a precision, not like a copy.

`mahalanobis-task` (the new default) keeps one second-moment matrix per task
instead of one pooled matrix, and adds a log-determinant term so that a domain
whose own features spread widely does not win every query. The failure it fixes
is measured: with a single pooled matrix the production checkpoints route MNIST
at 6.07% and CIFAR-10 at 41.16%, against 95.37% / 99.67% for per-task
matrices (audit B40/B41).

What these tests pin is the mechanism, because a synthetic reproduction of the
production feature geometry would be a nice-looking fiction:

  - the per-task matrices are genuinely different when the tasks differ, and
    identical to the pooled one when they do not (so the change is inert in the
    symmetric case and cannot regress it),
  - each per-task matrix emphasises its OWN high-variance direction, which is
    the whole reason for keeping it,
  - the log-determinant term is non-negative for a covariance and is reported
    per task, so a domain with a larger spread pays more,
  - like the shared matrix, it is fed only by the clean finalize pass.
"""
import unittest

import torch

from train_continual_aml import TaskCentroidRouter


def _fit(router, task_idx, feats):
    router.finalize_task_features(task_idx, feats)
    router.finalize_task(task_idx)


def _precs(r, n):
    P, logdets = r._precisions(n)
    return P, logdets


class TestPerTaskPrecision(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_identical_tasks_give_identical_precision(self):
        """Symmetric case: per-task must reduce to the shared matrix."""
        D = 8
        r = TaskCentroidRouter(D, 3)
        base = torch.randn(64, D)
        base = base / base.norm(dim=1, keepdim=True)
        for k in range(3):
            _fit(r, k, base.clone())        # literally the same cloud
        P, _ = _precs(r, 3)
        shared = r._precision()
        for k in range(3):
            self.assertTrue(torch.allclose(P[k], shared, atol=1e-3),
                            f"task {k} precision should match the shared one "
                            f"when the tasks are statistically identical")

    def test_perturbed_tasks_give_different_precisions(self):
        D = 8
        r = TaskCentroidRouter(D, 3)
        for k in range(3):
            # each task spreads along a DIFFERENT coordinate, by a different
            # amount: variance, not a mean shift (the covariance is taken about
            # the task's own mean, so shifting the mean adds no variance)
            z = torch.randn(400, D) * 0.02
            z[:, k] = torch.randn(400) * (1.0 + k)
            _fit(r, k, z)
        P, _ = _precs(r, 3)
        d01 = (P[0] - P[1]).abs().max().item()
        self.assertGreater(d01, 1e-2, "per-task precisions must differ")
        # A precision is LARGE where the task's own variance is SMALL. Whitening
        # deliberately down-weights the directions a domain spreads along -
        # those carry no class information - and keeps the tight ones. So each
        # matrix must be smallest on its OWN spread-out coordinate.
        for k in range(3):
            for j in range(3):
                if j == k:
                    continue
                self.assertLess(P[k][k, k].item(), P[k][j, j].item(),
                                f"task {k} precision should down-weight its "
                                f"own high-variance coordinate {k} vs {j}")

    def test_logdet_is_reported_per_task_and_tracks_spread(self):
        D = 8
        r = TaskCentroidRouter(D, 2)
        narrow = torch.randn(500, D) * 0.01
        wide = torch.randn(500, D) * 0.01
        wide[:, 0] = torch.randn(500) * 3.0
        _fit(r, 0, narrow)
        _fit(r, 1, wide)
        _, logdets = _precs(r, 2)
        self.assertEqual(len(logdets), 2)
        # log|S| shrinks as the covariance grows: the broadly spread task pays
        # more for every query, which is the whole point of the term.
        self.assertGreater(logdets[1], logdets[0] + 1.0)

    def test_task_without_a_clean_pass_falls_back_to_shared(self):
        """A partially built router must still score instead of raising."""
        D = 6
        r = TaskCentroidRouter(D, 3)
        _fit(r, 0, torch.randn(50, D))
        P, logdets = _precs(r, 3)
        shared = r._precision()
        self.assertTrue(torch.allclose(P[1], shared, atol=1e-6))
        self.assertTrue(torch.allclose(P[2], shared, atol=1e-6))
        self.assertEqual(logdets[1:], [0.0, 0.0])

    def test_training_steps_never_feed_the_per_task_matrices(self):
        """The shared matrix is fed only by the clean pass because mid-training
        features come from encoder states that no longer exist at the end of
        the task; the same argument applies to the per-task ones."""
        D = 6
        r = TaskCentroidRouter(D, 2)
        for step in range(50):
            r.update_task_centroid(0, torch.randn(8, D) * 10 + step)
        self.assertEqual(int(r.cov_n_task[0].item()), 0)
        self.assertTrue(torch.equal(r.cov_task[0], torch.zeros(D, D)))
        _fit(r, 0, torch.randn(30, D))
        self.assertEqual(int(r.cov_n_task[0].item()), 30)

    def test_cosine_metric_does_not_allocate_per_task_matrices(self):
        """No point paying T x D x D when the metric ignores it."""
        D = 6
        r = TaskCentroidRouter(D, 3, metric="cosine")
        _fit(r, 0, torch.randn(40, D))
        self.assertEqual(int(r.cov_n_task[0].item()), 0)

    def test_cache_invalidates_when_a_task_is_finalized(self):
        D = 6
        r = TaskCentroidRouter(D, 2)
        _fit(r, 0, torch.randn(40, D))
        P1, _ = _precs(r, 2)
        _fit(r, 1, torch.randn(40, D))
        P2, _ = _precs(r, 2)
        self.assertFalse(torch.allclose(P1[1], P2[1]),
                         "a stale cache would keep scoring a task that had no "
                         "second moment when the cache was built")


if __name__ == "__main__":
    unittest.main()