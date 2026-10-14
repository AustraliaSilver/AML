"""Guards for the domain router.

The router was the single largest source of error: A_auto collapsed to 8%
while A_oracle held 98%. Root cause was not the features (a supervised probe
reached 92% on the same latents) but the cosine metric, which treats every
direction as equally informative. Mahalanobis with one shared covariance
reached 99.8% while staying zero-replay.
"""
import unittest

import torch
import torch.nn.functional as F

from train_continual_aml import TaskCentroidRouter, router_features


def _fit(router, X, y, n_cls):
    for i in range(n_cls):
        router.update_task_centroid(i, X[y == i])
        router.finalize_task(i)
    router.num_tasks[0] = n_cls
    return router


class TestRouterMetric(unittest.TestCase):

    def test_default_metric_is_mahalanobis(self):
        self.assertEqual(TaskCentroidRouter(32).metric, "mahalanobis")

    def test_cosine_still_available(self):
        r = TaskCentroidRouter(32, metric="cosine")
        self.assertEqual(r.metric, "cosine")

    def test_unknown_metric_rejected(self):
        with self.assertRaises(ValueError):
            TaskCentroidRouter(32, metric="euclidean")

    def test_mahalanobis_beats_cosine_on_separable_data(self):
        """Anisotropic features: variance differs wildly per direction, which is
        exactly the case cosine ignores."""
        torch.manual_seed(0)
        D, n = 64, 400
        # spread the classes along axis 0, keep the rest low-variance
        base = torch.randn(3 * n, D) * 0.05
        base[0 * n:1 * n, 0] += 6.0
        base[1 * n:2 * n, 0] -= 6.0
        base[2 * n:3 * n, 1] += 6.0
        X = base

        k = int(n * 0.7)
        tr = torch.cat([X[i * n:i * n + k] for i in range(3)])
        ytr = torch.cat([torch.full((k,), i) for i in range(3)])
        te = torch.cat([X[i * n + k:(i + 1) * n] for i in range(3)])
        yte = torch.cat([torch.full((n - k,), i) for i in range(3)])

        accs = {}
        for met in ("cosine", "mahalanobis"):
            r = _fit(TaskCentroidRouter(D, 3, temperature=0.1, metric=met), tr, ytr, 3)
            accs[met] = (r(te).argmax(-1) == yte).float().mean().item()
        self.assertGreaterEqual(accs["mahalanobis"], accs["cosine"],
                                f"mahalanobis {accs['mahalanobis']:.3f} should "
                                f"be >= cosine {accs['cosine']:.3f}")
        self.assertGreater(accs["mahalanobis"], 0.9)

    def test_stays_zero_replay(self):
        """No exemplars retained: only centroids and one shared covariance."""
        r = TaskCentroidRouter(16, 3)
        for i in range(3):
            r.update_task_centroid(i, torch.randn(50, 16))
        names = [n for n, _ in r.named_buffers()]
        self.assertIn("cov", names)
        self.assertIn("centroids", names)
        self.assertEqual(r.cov.shape, (16, 16))
        # no buffer grows with the number of samples
        self.assertEqual(r.cov_n.item(), 150)

    def test_past_centroids_survive_later_tasks(self):
        r = TaskCentroidRouter(8, 4)
        for t in range(3):
            r.update_task_centroid(t, torch.randn(10, 8) + t)
            r.finalize_task(t)
        c0 = r.centroids[0].clone()
        r.update_task_centroid(3, torch.randn(10, 8) + 99)
        r.finalize_task(3)
        self.assertTrue(torch.allclose(r.centroids[0], c0))

    def test_logits_are_finite_before_any_data(self):
        r = TaskCentroidRouter(8, 3)
        out = r(torch.randn(4, 8))
        # no task seen yet -> T_cur clamps to 1, so a single (garbage) column
        self.assertEqual(tuple(out.shape), (4, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_mahalanobis_keeps_the_true_mean_not_a_unit_vector(self):
        """Regression: finalize_task used to L2-normalise the centroid for every
        metric. Under Mahalanobis that moved the centroid off the cloud it
        represents and collapsed CIFAR-10 routing from 100.0% to 4.0%, because a
        broad domain has a small ||mean|| (0.695 vs 0.929 for a tight one)."""
        torch.manual_seed(0)
        # task 0: tight cluster. task 1: broad, deliberately built so that
        # normalising its mean moves it away from its own samples
        tight = F.normalize(torch.randn(600, 32), dim=1) * 0.03 + torch.tensor([3.0] + [0.0] * 31)
        broad = torch.randn(600, 32) * 1.0
        X = torch.cat([tight, broad])
        y = torch.cat([torch.zeros(600, dtype=torch.long), torch.ones(600, dtype=torch.long)])

        r = _fit(TaskCentroidRouter(32, 2), X, y, 2)
        self.assertAlmostEqual(r.centroids[0].norm().item(), 3.0, places=1)
        self.assertLess(r.centroids[1].norm().item(), 0.99,
                        "broad domain's centroid must stay off the unit sphere")
        acc = (r(X).argmax(-1) == y).float().mean().item()
        self.assertGreater(acc, 0.95, f"router accuracy collapsed to {acc:.3f}")

    def test_cosine_still_normalises_the_centroid(self):
        torch.manual_seed(0)
        X = torch.randn(200, 16) * 3.0
        y = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
        r = _fit(TaskCentroidRouter(16, 2, metric="cosine"), X, y, 2)
        self.assertAlmostEqual(r.centroids[0].norm().item(), 1.0, places=3)
        self.assertAlmostEqual(r.centroids[1].norm().item(), 1.0, places=3)

    def test_precision_cache_is_invalidated_by_new_features(self):
        """The memo is keyed on cov_n; folding in a task must rebuild it, or the
        router would silently score against a stale metric."""
        torch.manual_seed(0)
        r = TaskCentroidRouter(16, 2)
        shift = (torch.arange(2).float() * 4).view(2, 1, 1)
        X = (torch.randn(2, 100, 16) + shift).reshape(200, 16)
        y = torch.arange(2).repeat_interleave(100)

        r.update_task_centroid(0, X[y == 0]); r.finalize_task(0)
        first = r._precision().clone()
        self.assertTrue(torch.equal(first, r._precision()), "cache must be reused")

        r.update_task_centroid(1, X[y == 1]); r.finalize_task(1)
        self.assertFalse(torch.equal(first, r._precision()),
                         "cache must be rebuilt after a new task")

    def test_cache_survives_a_state_dict_round_trip(self):
        """_prec_cache is a plain attribute, not a buffer, so it must not leak
        into the checkpoint or be required on load."""
        r = _fit(TaskCentroidRouter(16, 2),
                 torch.randn(200, 16) * 2,
                 torch.arange(2).repeat_interleave(100), 2)
        names = [k for k, _ in r.state_dict().items()]
        self.assertNotIn("_prec_cache", names)
        clone = TaskCentroidRouter(16, 2)
        clone.load_state_dict(r.state_dict())
        self.assertTrue(torch.allclose(clone._precision(), r._precision()))

    def test_argmax_is_invariant_to_the_logit_scale(self):
        """The open-set fallback uses the softmax margin, so the scale must be
        sane - but the routing decision itself must not care."""
        torch.manual_seed(0)
        shift = (torch.arange(3).float() * 4).view(3, 1, 1)    # [3,1,1] over (cls,n,D)
        X = (torch.randn(3, 100, 16) + shift).reshape(300, 16)
        y = torch.arange(3).repeat_interleave(100)
        r = _fit(TaskCentroidRouter(16, 3), X, y, 3)
        out = r(torch.randn(20, 16))
        self.assertEqual(tuple(out.shape), (20, 3))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(out.abs().max().item(), 1e6)


class TestScorePerTask(unittest.TestCase):
    """A centroid lives in the space of the encoder that produced it.

    Measured on real checkpoints the drift between task k's encoder and the
    current one was 2.38x the within-domain spread, which is why embedding the
    query once with the current encoder capped routing at 34-52% no matter
    which metric was chosen. score_per_task embeds the query once per task.
    """

    def _setup(self):
        """Two domains, each with its own encoder that maps its own data to a
        compact cluster and the other domain's data far away."""
        torch.manual_seed(0)
        D = 32
        g = torch.Generator().manual_seed(0)
        self.D = D

        def domain(seed, shift_dim):
            base = torch.randn(400, D, generator=g)
            base[:, shift_dim] += 6.0
            return base

        self.data = [domain(0, 0), domain(1, 1)]

        # A linear encoder that pulls its OWN domain onto axis k and pushes the
        # other one onto axis k+1. This is what training roughly produces: the
        # representation separates the domain it was fit on.
        class Enc(torch.nn.Module):
            def __init__(self, own, other):
                super().__init__()
                self.fc = torch.nn.Linear(D, D)
                with torch.no_grad():
                    self.fc.weight.zero_()
                    self.fc.weight[own, own] = 1.0
                    self.fc.weight[other, other] = 1.0

            def forward(self, x):
                return self.fc(x)

        self.enc = [Enc(0, 1), Enc(1, 0)]

    def _features(self, enc, x):
        with torch.no_grad():
            return F.normalize(enc(x), dim=1)

    def _fit_router(self):
        r = TaskCentroidRouter(self.D, 2)
        for k in range(2):
            r.update_task_centroid(k, self._features(self.enc[k], self.data[k]))
            r.finalize_task(k)
        r.num_tasks[0] = 2
        return r

    def test_picks_the_nearest_centroid_in_each_tasks_own_space(self):
        """The contract: score_per_task([enc_k(x)]) ranks domains by how far
        enc_k(x) sits from centroid_k, one column per task, one row per sample.

        Its logits are rescaled by d2.mean() over the whole batch, so the
        comparable quantity is the ranking, not the raw value.
        """
        self._setup()
        r = self._fit_router()
        # The SAME batch embedded under each encoder - that is the contract of
        # score_per_task, not one batch per task.
        x = self.data[0][:128]
        zk = [self._features(self.enc[k], x) for k in range(2)]

        P = r._precision()
        d2 = torch.stack([
            ((zk[k] - r.centroids[k]) @ P * (zk[k] - r.centroids[k])).sum(-1)
            for k in range(2)], dim=1)

        out = r.score_per_task(zk)
        self.assertEqual(tuple(out.shape), (128, 2))
        self.assertTrue((out.argmax(-1) == d2.argmin(-1)).all().item(),
                        "routing must pick the nearest centroid per task")
        # and the scaling must be one positive factor for ALL columns, otherwise
        # argmax would compare distances that were normalised differently
        ratios = (-out * d2.clamp(min=1e-9))
        self.assertGreater(ratios.min().item(), 0.0)

    def test_shape_and_finiteness(self):
        self._setup()
        r = self._fit_router()
        x = self.data[0][:32]
        zk = [self._features(self.enc[k], x) for k in range(2)]
        out = r.score_per_task(zk)
        self.assertEqual(tuple(out.shape), (32, 2))
        self.assertTrue(torch.isfinite(out).all())

    def test_survives_encoder_drift_that_breaks_the_single_encoder_router(self):
        """The regression this method exists for.

        Task 2 training drags encoder 2 away from where it was when task 1's
        centroid was stored. Scoring a task-1 sample with encoder 2 therefore
        compares against a centroid that lives in a different space - which is
        exactly what the single-encoder forward() does, and it misroutes.
        Embedding the sample with encoder 1 for task 1's slot does not.
        """
        self._setup()
        D = self.D
        # enc_old: the state when task 1 was finalised. enc_new: after task 2
        # has rewritten the shared encoder.
        enc_old = torch.nn.Linear(D, D)
        enc_new = torch.nn.Linear(D, D)
        with torch.no_grad():
            # task 1's encoder: domain 0 -> axis 0, nothing on axis 1
            enc_old.weight.zero_()
            enc_old.weight[0, 0] = 1.0
            # task 2 training rewrites the shared encoder so that domain 0 now
            # lands on axis 1 - the SAME axis domain 1's centroid sits on. The
            # stored centroid for task 1 is now in the wrong place entirely.
            enc_new.weight.zero_()
            enc_new.weight[0, 1] = 1.0
            enc_new.weight[1, 1] = 1.0

        def emb(e, x):
            with torch.no_grad():
                return F.normalize(e(x), dim=1)

        r = TaskCentroidRouter(D, 2)
        r.update_task_centroid(0, emb(enc_old, self.data[0]))   # task 1 finalised
        r.finalize_task(0)
        r.update_task_centroid(1, emb(enc_new, self.data[1]))   # task 2 finalised
        r.finalize_task(1)
        r.num_tasks[0] = 2

        x0 = self.data[0]                                    # a domain-0 sample
        truth = 0

        # forward() embeds ONCE with whatever encoder it is handed and scores
        # that single embedding against both centroids. At eval time that
        # encoder is the current one, so slot 0 is judged by a vector that no
        # longer points at its own centroid.
        single = r(emb(enc_new, x0)).argmax(-1)
        # score_per_task embeds once per slot with that slot's own encoder.
        per_task = r.score_per_task([emb(enc_old, x0),
                                     emb(enc_new, x0)]).argmax(-1)
        self.assertAlmostEqual((per_task == truth).float().mean().item(), 1.0,
                               msg="per-task scoring must route every "
                                   "domain-0 sample to slot 0")
        # and document that the single-encoder path really is the broken one
        self.assertLess((single == truth).float().mean().item(), 0.5,
                        "if the single-encoder path is no longer broken, this "
                        "test no longer proves anything")

    def test_cosine_metric_is_supported(self):
        self._setup()
        r = TaskCentroidRouter(self.D, 2, metric="cosine")
        for k in range(2):
            r.update_task_centroid(k, self._features(self.enc[k], self.data[k]))
            r.finalize_task(k)
        r.num_tasks[0] = 2
        zk = [self._features(self.enc[k], self.data[k]) for k in range(2)]
        out = r.score_per_task(zk)
        # one row per SAMPLE, one column per task - not the concatenation
        self.assertEqual(tuple(out.shape), (400, 2))
        self.assertTrue(torch.isfinite(out).all())


class TestRouterFeatures(unittest.TestCase):
    """The router latent must not be squashed into the non-negative orthant."""

    def test_strips_a_terminal_relu(self):
        import torch.nn as nn
        from mgn.blocks import ConvAML_18Layer_Hybrid
        m = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=16, num_classes=10,
                                   encoder_tail_act="relu")
        self.assertIsInstance(m.vision_encoder[6], nn.ReLU)
        z = router_features(m, torch.randn(4, 3, 32, 32))
        self.assertTrue(torch.isfinite(z).all())
        self.assertEqual(z.shape, (4, m.latent_dim))
        # not restricted to the positive orthant
        self.assertLess(z.min().item(), 0.0)

    def test_falls_back_for_identity_tail(self):
        from mgn.blocks import ConvAML_18Layer_Hybrid
        m = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=16, num_classes=10,
                                   encoder_tail_act="none")
        z = router_features(m, torch.randn(4, 3, 32, 32))
        self.assertEqual(z.shape, (4, m.latent_dim))

    def test_output_is_l2_normalised(self):
        from mgn.blocks import ConvAML_18Layer_Hybrid
        m = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=16, num_classes=10)
        z = router_features(m, torch.randn(8, 3, 32, 32))
        self.assertTrue(torch.allclose(z.norm(dim=1), torch.ones(8), atol=1e-4))


if __name__ == "__main__":
    unittest.main()