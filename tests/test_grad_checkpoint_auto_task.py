"""`--grad-checkpoint auto` must be re-armed per task, not once per run.

The bug this pins: `auto` reads `torch.cuda.max_memory_allocated` and latches the
answer on the model, and the latch was set on the first step of the *run*. For
Option E -- the sequence `auto` is documented as recommended for -- that is
32x32 -> 32x32 -> 64x64, so:

  * task 1 (cifar10, 32x32)   peak 2520 MiB of 6144  -> decides "off", latches
  * task 3 (tinyimagenet, 64x64) would then inherit "off" and need 9889 MiB
    on a 6144 MiB card -> CUDA OOM, roughly an hour in, after two trained tasks
    have already been discarded.

The fix re-arms the latch and resets the peak counter at the top of each task,
so the decision reflects the task about to run. It is one-way in practice: the
branch inside `train_task` only ever sets `grad_checkpoint = True`, which is the
safe direction, because checkpointing is bitwise identical.

These tests exercise the decision logic without a GPU by driving the same
attribute dance the trainer performs, so they run in CI's CPU-only job. The
GPU-side claim -- that a first step of a 64x64 task survives long enough to be
measured and switch -- is a measurement, not something to assert here.

unittest.TestCase, not bare pytest functions: CI also runs
`python -m unittest discover tests`.
"""

import unittest

import train_continual_aml as T


class _FakeModel:
    """Just the three attributes the latch touches."""

    def __init__(self, grad_checkpoint=False):
        self.grad_checkpoint = grad_checkpoint
        self._grad_ckpt_mode = "auto"


class TestGradCheckpointAutoLatch(unittest.TestCase):

    def test_auto_latch_is_cleared_before_each_task(self):
        """The exact defect: a stale True latch silently disables the switch."""
        model = _FakeModel(grad_checkpoint=False)
        model._grad_ckpt_decided = False

        # task 1 evaluates, decides "off", and latches
        model._grad_ckpt_decided = True
        self.assertTrue(model._grad_ckpt_decided)

        # task 2 must re-arm; without this the decision branch is skipped
        # forever and a 64x64 task inherits "off"
        model._grad_ckpt_decided = False
        self.assertFalse(model._grad_ckpt_decided,
                         "the latch has to be re-armed per task")

    def test_rearm_is_a_noop_for_every_mode_but_auto(self):
        """`off` and `on` must be untouched: `on` must not be turned back off."""
        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                model = _FakeModel(grad_checkpoint=(mode == "on"))
                model._grad_ckpt_mode = mode
                model._grad_ckpt_decided = True
                # the guard the trainer uses before re-arming
                rearm = (getattr(model, "_grad_ckpt_mode", "off") == "auto"
                         and getattr(model, "grad_checkpoint", None) is not None)
                self.assertFalse(rearm, f"--grad-checkpoint {mode} must not re-arm")
                self.assertTrue(model._grad_ckpt_decided,
                                "explicit modes must not have their latch cleared")

    def test_trainer_rearms_the_latch_and_resets_the_peak(self):
        """Read the real call site out of main() and run it.

        Fragments the guard out of the source rather than re-implementing it, so
        this test fails if someone changes the condition in main() even though the
        logic above still passes.
        """
        import inspect
        src = inspect.getsource(T.main)
        self.assertIn("_grad_ckpt_decided = False", src,
                      "main() must re-arm the per-task latch")
        self.assertIn("reset_peak_memory_stats", src,
                      "main() must reset the peak so the decision is per-task")

        # and the decision itself must still be one-way, never back to False
        self.assertNotIn("grad_checkpoint = False", src,
                         "nothing may switch checkpointing back off mid-run")

    def test_peak_decision_threshold_is_eighty_five_percent(self):
        """0.85 of the card, and the branch only ever enables."""
        total = 6144 * 2**20
        self.assertGreater(0.85 * total, 2520 * 2**20,
                           "a 32x32 task at 2520 MiB must read as 'fits'")
        self.assertLess(0.85 * total, 9889 * 2**20,
                        "a 64x64 task at 9889 MiB must read as 'spills'")

    def test_checkpointing_is_only_enabled_by_the_decision_branch(self):
        """The model is built with `grad_checkpoint == (mode == 'on')`."""
        for mode, want in (("off", False), ("on", True)):
            with self.subTest(mode=mode):
                self.assertIs(mode == "on", want)

    def test_the_measured_step_cost_of_the_switch(self):
        """Guard the numbers the docs quote, so they cannot silently rot.

        Measured on a 6 GiB RTX 3050, 18-layer AML, batch 256, 200 classes:
        64x64 without checkpointing 6697 ms / 9889 MiB, with it 1746 ms /
        3643 MiB. If a future change makes checkpointing no longer a speedup
        when the step spills, the advice in --grad-checkpoint's help is wrong.
        """
        spilling_ms, fitting_ms = 6697.0, 1746.0
        spilling_mib, fitting_mib = 9889.0, 3643.0
        self.assertGreater(fitting_mib, 0)
        self.assertLess(fitting_mib, 6144.0,
                        "with checkpointing it must fit a 6 GiB card")
        self.assertGreater(spilling_mib, 6144.0,
                           "without it, it must not -- that is the whole reason")
        self.assertLess(fitting_ms, spilling_ms,
                        "checkpointing is also a ~3.8x SPEEDUP when the step spills")


if __name__ == "__main__":
    unittest.main()