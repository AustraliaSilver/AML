"""The default run must be the zero-replay run that the docs describe.

Every headline number in the README and the paper is a **zero-replay** result.
`--replay-per-class` defaulted to **400**, i.e. 4 000 stored exemplars plus a
consolidation pass over them, so a bare `python train_continual_aml.py` silently
measured a *different method* from the documented `--replay-per-class 0`
command. Nothing crashed; the two runs simply disagree, which is the worst kind
of default bug: the reproduction command and the headline claim drift apart with
no error anywhere.

The test asserts against `build_arg_parser()` - the trainer's own parser, not a
copy of the defaults - so it tracks the code.
"""
import subprocess
import sys
import unittest

from train_continual_aml import build_arg_parser


class TestZeroReplayIsTheDefault(unittest.TestCase):

    def test_replay_default_is_zero(self):
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.replay_per_class, 0,
                         "the default run must be the zero-replay run the docs "
                         "claim to have measured")

    def test_explicit_replay_is_still_available(self):
        args = build_arg_parser().parse_args(["--replay-per-class", "5"])
        self.assertEqual(args.replay_per_class, 5,
                         "replay must stay reachable for the comparison runs")

    def test_norm_default_is_instance_norm(self):
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.norm, "in")
        self.assertEqual(args.encoder_tail_act, "none")
        self.assertEqual(args.ewc_lambda, 0.0,
                         "EWC is disabled by default; it was refuted at scale "
                         "(and re-measuring it is still open)")

    def test_help_states_the_replay_default_change(self):
        r = subprocess.run([sys.executable, "train_continual_aml.py", "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        i = r.stdout.rindex("--replay-per-class")   # the detailed entry, not the usage line
        self.assertIn("DEFAULT CHANGED 400 -> 0", r.stdout[i:i + 900])


if __name__ == "__main__":
    unittest.main()