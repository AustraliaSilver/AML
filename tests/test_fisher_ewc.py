"""FisherEWC: the exact multi-task quadratic, stored in O(1).

EWC's penalty is

    L(theta) = (lambda/2) * sum_t sum_i F_i^(t) (theta_i - theta*_i^(t))^2

The naive O(1) shortcut is to keep a running sum of F alongside a single
re-pinned theta*, which is what the code did. That is wrong: after task 2 it
scores task 1's Fisher against task 2's weights, anchoring theta to a point
that had already forgotten task 1.

The exact expansion is

    sum_t F^(t)(theta - theta*^(t))^2
        = (sum_t F^(t)) theta^2 - 2 (sum_t F^(t) theta*^(t)) theta + const

and the constant does not depend on theta, so it never enters the gradient.
Two accumulators - A = sum F, B = sum F*theta* - are therefore both exact and
the same size as the (F, theta*) pair they replace.

These tests check the expansion against a brute-force T-copy reference, and
assert that the minimum sits at the Fisher-weighted mean of every task's
optimum rather than at the most recent one, which is the defect itself.
"""
import unittest

import torch
import torch.nn as nn

from train_continual_aml import FisherEWC


class _Tiny(nn.Module):
    """One weight tensor, so the algebra can be checked element-wise."""

    def __init__(self, d=6):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d))

    def forward(self, x):
        return self.w @ x


def _run_task(ewc, model, drift):
    """One synthetic task: move theta, record a gradient, close the window.

    The drift matters. With a stationary theta every task's optimum is the same
    vector and the difference between the broken and the exact form is
    invisible - which is exactly why the original defect survived.
    """
    model.w.data.add_(drift)
    g = torch.randn_like(model.w)
    for p in model.parameters():
        p.grad = g.detach().clone()
    ewc._n = 1
    ewc._fisher = {n: p.grad.detach().float() ** 2
                   for n, p in model.named_parameters()}
    ewc.consolidate()
    return g ** 2, model.w.detach().clone()


class TestExactQuadratic(unittest.TestCase):

    def _replay(self, n_tasks=3, lam=1.0, seed=3):
        torch.manual_seed(seed)
        model = _Tiny()
        ewc = FisherEWC(model, ewc_lambda=lam)
        fishers, stars = [], []
        for _ in range(n_tasks):
            f, s = _run_task(ewc, model, torch.randn_like(model.w))
            fishers.append(f)
            stars.append(s)
        return model, ewc, fishers, stars

    def test_matches_brute_force_multi_task_reference(self):
        model, ewc, fishers, stars = self._replay()
        theta = torch.randn_like(model.w)
        model.w.data.copy_(theta)

        ref = sum(float((f * (theta - s) ** 2).sum()) for f, s in zip(fishers, stars))

        # penalty() is the completed square A(theta - B/A)^2, so it is the true
        # quadratic minus the theta-independent spread of the tasks' optima
        # about their Fisher-weighted mean.
        A, B = ewc._fisher_all["w"], ewc._star_all["w"]
        mu = B / A
        spread = sum(float((f * (s - mu) ** 2).sum()) for f, s in zip(fishers, stars))
        # lambda = 1 here, so 2*penalty is the bare quadratic sum
        self.assertAlmostEqual(2.0 * float(ewc.penalty()) + spread, ref, places=2)

    def test_accumulators_match_the_explicit_sums(self):
        _, ewc, fishers, stars = self._replay()
        A = sum(fishers)
        B = sum(f * s for f, s in zip(fishers, stars))
        self.assertTrue(torch.allclose(ewc._fisher_all["w"], A, atol=1e-5))
        self.assertTrue(torch.allclose(ewc._star_all["w"], B, atol=1e-5))

    def test_gradient_matches_the_brute_force_reference(self):
        model, ewc, fishers, stars = self._replay(lam=3.0)
        theta = torch.randn_like(model.w)
        model.w.data.copy_(theta)
        # autograd.grad, not backward: .grad still holds the synthetic task
        # gradient from _run_task and backward() would accumulate onto it.
        got = torch.autograd.grad(ewc.penalty(), model.w)[0]

        # reference gradient of sum_t F^(t)(theta - theta*^(t))^2
        th = theta.clone().requires_grad_(True)
        total = sum((f * (th - s) ** 2).sum() for f, s in zip(fishers, stars))
        ref = torch.autograd.grad(0.5 * 3.0 * total, th)[0]
        self.assertTrue(torch.allclose(got, ref, atol=1e-4))

    def test_minimum_is_the_fisher_weighted_mean_not_the_last_task(self):
        """The defect in one assertion: with the broken scheme the penalty was
        minimised at theta*^(last). With the exact form it is minimised at
        B/A, the Fisher-weighted average of every task's optimum."""
        torch.manual_seed(2)
        model = _Tiny()
        ewc = FisherEWC(model, ewc_lambda=1.0)
        snaps, scales = [], (5.0, 0.2)
        for t in range(2):
            # task 0 gets a large Fisher, so its optimum must dominate
            model.w.data.add_(torch.randn_like(model.w) * 2.0)
            g = torch.randn_like(model.w) * scales[t]
            for p in model.parameters():
                p.grad = g.detach().clone()
            ewc._n = 1
            ewc._fisher = {n: p.grad.detach().float() ** 2
                           for n, p in model.named_parameters()}
            ewc.consolidate()
            snaps.append(model.w.detach().clone())

        A, B = ewc._fisher_all["w"], ewc._star_all["w"]
        optimum = B / A
        d_first = float((optimum - snaps[0]).norm())
        d_last = float((optimum - snaps[1]).norm())
        self.assertLess(d_first, d_last,
                        "penalty must anchor task 0 to its OWN optimum, not to "
                        "the most recent task's weights")

    def test_broken_scheme_would_have_been_wrong(self):
        """Quantify the defect rather than assert it in prose: the old
        (F_total, theta*_last) penalty, evaluated at the exact optimum, is
        strictly worse than the true multi-task quadratic."""
        model, ewc, fishers, stars = self._replay(seed=7)
        A, B = ewc._fisher_all["w"], ewc._star_all["w"]
        optimum = B / A
        star_last = stars[-1]

        true_pen = sum(float((f * (optimum - s) ** 2).sum())
                       for f, s in zip(fishers, stars))
        broken_pen = sum(float((f * (optimum - star_last) ** 2).sum())
                         for f in fishers)
        self.assertGreater(broken_pen, true_pen,
                           "anchoring every task to the last optimum is worse")

    def test_memory_is_two_vectors_per_parameter(self):
        torch.manual_seed(0)
        model = _Tiny(d=64)
        ewc = FisherEWC(model, ewc_lambda=1.0)
        _run_task(ewc, model, torch.randn_like(model.w))
        n = model.w.numel()
        self.assertEqual(ewc.state_bytes(), 2 * n,
                         "A and B cost exactly what (F, theta*) cost")


if __name__ == "__main__":
    unittest.main()