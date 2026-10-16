"""Guards for `--grad-checkpoint` / `grad_checkpoint=` on the hybrid model.

Activation checkpointing is only worth having if it is a *pure memory* trade. If it
changed the numbers it would be a research change and would need a full run; the
whole reason it can be enabled by a heuristic mid-run (`--grad-checkpoint auto`) is
that it does not. These tests pin that:

  * it is off by default, and off means the ordinary call (no `checkpoint` at all);
  * with it on, `torch.utils.checkpoint.checkpoint` is genuinely reached, once per
    executed layer;
  * one seeded train-mode forward+backward gives bitwise-identical output, routing
    and every gradient;
  * **three chained training steps from a single seed give identical losses and
    identical final weights.** This is the test that matters: the trainer seeds
    once, then each step draws its architecture from the global RNG (Gumbel over
    `pi_logits`, Gumbel over the block-mode logits, `0.05 * randn` on the depth
    gates). If a checkpointed block perturbed the RNG stream -- which is what
    `torch.random.fork_rng` inside `checkpoint` could do -- step 2 would sample a
    different architecture and the run would diverge. It does not.
  * it is not entered under `torch.no_grad()`, where it would be pure overhead and
    would warn that no input requires grad.

unittest.TestCase, not bare pytest functions: CI runs
`python -m unittest discover tests` and does not collect pytest-style functions.
"""
import unittest

import torch

import mgn.blocks.hybrid_18 as h18
from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid


def _model(grad_checkpoint=False, num_classes=4):
    torch.manual_seed(11)
    return ConvAML_18Layer_Hybrid(
        K=4, latent_dim=32, max_layers=4, max_dim=16, num_classes=num_classes,
        num_heads=2, norm_type="gn1", encoder_tail_act="none",
        grad_checkpoint=grad_checkpoint)


def _fixed_x(shape=(3, 3, 16, 16), seed=123):
    """Input drawn from its own generator, so global RNG order cannot change it.

    (Drawing it from the global stream makes the comparison depend on how much RNG
    the *previous* model consumed -- an easy way to write a test that fails for the
    wrong reason.)"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _run(model, x, seed=5):
    """One seeded train-mode forward+backward; returns output, routing, grads."""
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    out, _struct, pi, _diag = model(x, tau=0.8)
    out.float().pow(2).mean().backward()
    grads = {k: p.grad.detach().clone()
             for k, p in model.named_parameters() if p.grad is not None}
    return out.detach().clone(), pi.detach().clone(), grads


def _trajectory(grad_checkpoint, steps=3):
    """N training steps from ONE seed at the start; returns losses and weights."""
    model = _model(grad_checkpoint)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = _fixed_x()
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x, tau=0.8)[0]
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return losses, state


class _CheckpointSpy:
    """Count calls to `torch.utils.checkpoint.checkpoint` from the real model."""

    def __enter__(self):
        self.n = 0
        self._orig = h18._ckpt.checkpoint

        def spy(fn, *args, **kwargs):
            self.n += 1
            return self._orig(fn, *args, **kwargs)

        h18._ckpt.checkpoint = spy
        return self

    def __exit__(self, *exc):
        h18._ckpt.checkpoint = self._orig
        return False


class TestGradCheckpointDefaults(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(_model().grad_checkpoint)

    def test_not_reached_when_off(self):
        model = _model(grad_checkpoint=False)
        with _CheckpointSpy() as spy:
            _run(model, _fixed_x())
        self.assertEqual(spy.n, 0, "checkpointing must not be entered when off")

    def test_reached_when_on(self):
        model = _model(grad_checkpoint=True)
        with _CheckpointSpy() as spy:
            _run(model, _fixed_x())
        self.assertGreaterEqual(spy.n, 4, "one checkpoint per executed layer")

    def test_not_reached_under_no_grad(self):
        """No grad -> nothing to save, so it must not be entered (and must not warn)."""
        model = _model(grad_checkpoint=True).eval()
        with _CheckpointSpy() as spy:
            with torch.no_grad():
                model(_fixed_x(), tau=0.8)
        self.assertEqual(spy.n, 0)


class TestGradCheckpointIdentity(unittest.TestCase):
    """The claim `--grad-checkpoint auto` rests on: same numbers, less memory."""

    def test_output_routing_and_gradients_bitwise_identical(self):
        off = _model(grad_checkpoint=False)
        on = _model(grad_checkpoint=True)
        x = _fixed_x()

        out_off, pi_off, g_off = _run(off, x)
        out_on, pi_on, g_on = _run(on, x)

        self.assertTrue(torch.equal(out_off, out_on),
                        "checkpointing changed the forward output")
        self.assertTrue(torch.equal(pi_off, pi_on),
                        "checkpointing changed the routing distribution")
        self.assertEqual(set(g_off), set(g_on),
                         "checkpointing changed which parameters got gradients")
        for k in g_off:
            self.assertTrue(torch.equal(g_off[k], g_on[k]),
                            f"checkpointing changed the gradient of {k}")
        self.assertGreater(len(g_off), 30, "sanity: the trunk should have grads")

    def test_flag_can_be_toggled_between_calls_safely(self):
        """`auto` flips the attribute at run time; that must be safe."""
        model = _model(grad_checkpoint=False)
        x = _fixed_x()
        out_a, _, _ = _run(model, x)
        model.grad_checkpoint = True
        out_b, _, _ = _run(model, x)
        model.grad_checkpoint = False
        out_c, _, _ = _run(model, x)
        self.assertTrue(torch.equal(out_a, out_b))
        self.assertTrue(torch.equal(out_a, out_c))


class TestGradCheckpointTrajectory(unittest.TestCase):
    """One seed, N steps: the RNG stream must not be perturbed by checkpointing."""

    def test_three_chained_steps_identical(self):
        torch.manual_seed(99)
        losses_off, weights_off = _trajectory(False)
        torch.manual_seed(99)
        losses_on, weights_on = _trajectory(True)

        self.assertEqual(losses_off, losses_on,
                         "the two runs diverged -- checkpointing moved the RNG stream")
        self.assertEqual(set(weights_off), set(weights_on))
        for k in weights_off:
            self.assertTrue(torch.equal(weights_off[k], weights_on[k]),
                            f"weights diverged at {k}")
        self.assertGreater(losses_off[0], 0.0)


class TestGradCheckpointFlag(unittest.TestCase):
    """The default must be `off`, so the shipped protocol stays bit-identical."""

    def _parser(self):
        from train_continual_aml import build_arg_parser
        return build_arg_parser()

    def test_parser_default_is_off(self):
        self.assertEqual(self._parser().parse_args([]).grad_checkpoint, "off")

    def test_accepts_on_and_auto(self):
        for mode in ("on", "auto"):
            self.assertEqual(
                self._parser().parse_args(["--grad-checkpoint", mode]).grad_checkpoint,
                mode)

    def test_rejects_anything_else(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["--grad-checkpoint", "sometimes"])


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestGradCheckpointOnTheProductionPath(unittest.TestCase):
    """Same trajectory check, on the real config: bf16 autocast, `norm=in`, 64x64.

    The CPU test above uses fp32/gn1/16x16; this is the path that actually runs on
    the GPU, where autocast and cuDNN are also in play.
    """

    def _prod_model(self, grad_checkpoint):
        torch.manual_seed(7)
        return ConvAML_18Layer_Hybrid(
            K=12, max_dim=48, max_layers=18, num_classes=10, norm_type="in",
            encoder_tail_act="none", grad_checkpoint=grad_checkpoint).cuda()

    def _traj(self, grad_checkpoint, steps=2):
        model = self._prod_model(grad_checkpoint).train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        g = torch.Generator(device="cpu").manual_seed(123)
        x = torch.randn(8, 3, 64, 64, generator=g).cuda()
        losses = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, tau=1.0)[0]
                loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return losses, state

    def test_two_steps_identical_in_bf16(self):
        torch.manual_seed(99)
        losses_off, weights_off = self._traj(False)
        torch.manual_seed(99)
        losses_on, weights_on = self._traj(True)
        self.assertEqual(losses_off, losses_on)
        for k in weights_off:
            self.assertTrue(torch.equal(weights_off[k], weights_on[k]),
                            f"weights diverged at {k}")


if __name__ == "__main__":
    unittest.main()
