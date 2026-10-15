"""The CL metrics printed at the end of a run must match their definitions.

`A_avg`, BWT and forgetting are the three numbers every table in the README and
the paper quotes, and they are the three numbers most easily transposed: the
trainer returns them as `(bwt, forgetting, A_avg)`, BWT is signed, forgetting is
a max over the *pre-final* stages only, and `A_avg` is the mean of the final
ROW (not of the diagonal). A silent mix-up does not crash anything - it just
makes the headline wrong, which is why the trainer's own function is pinned
here against an independent reference on matrices with hand-checked values.
"""
import unittest


def reference(M):
    """Straightforward definitions, written independently of the trainer."""
    T = len(M)
    a_avg = sum(M[T - 1]) / T
    bwt = sum(M[T - 1][j] - M[j][j] for j in range(T - 1)) / (T - 1)
    forg = [max(M[t][j] for t in range(j, T - 1)) - M[T - 1][j]
            for j in range(T - 1)]
    return a_avg, bwt, sum(forg) / len(forg)


def _trainer_cl_metrics(M):
    """Read `cl_metrics` out of main() and run it on M.

    It is defined inside main() and closes over main()'s ``T``, so it is
    extracted with ast rather than imported: copying it into this file would let
    the two drift, which is the failure mode being guarded against.
    """
    import ast
    import inspect
    import textwrap

    import train_continual_aml

    src = inspect.getsource(train_continual_aml.main)
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cl_metrics":
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         "<cl_metrics>", "exec"), ns)
            ns["T"] = len(M)
            return ns["cl_metrics"](M)
    raise AssertionError("cl_metrics not found in main()")


class TestCLMetrics(unittest.TestCase):

    MATRICES = [
        [[0.90, None, None], [0.70, 0.60, None], [0.65, 0.55, 0.50]],
        [[0.98, None, None], [0.97, 0.68, None], [0.96, 0.66, 0.60]],
        [[0.10, None, None], [0.20, 0.30, None], [0.40, 0.50, 0.60]],   # positive BWT
        [[0.90, None, None], [0.50, 0.90, None], [0.10, 0.10, 0.90]],   # big forgetting
    ]

    def test_matches_reference_definitions(self):
        for M in self.MATRICES:
            bwt, forg, a_avg = _trainer_cl_metrics(M)
            exp_avg, exp_bwt, exp_forg = reference(M)
            self.assertAlmostEqual(a_avg, exp_avg, places=9)
            self.assertAlmostEqual(bwt, exp_bwt, places=9)
            self.assertAlmostEqual(forg, exp_forg, places=9)

    def test_known_values(self):
        """The trainer returns (bwt, forgetting, A_avg). For the first matrix:
        BWT = ((0.70-0.90) + (0.55-0.60))/2 = -0.15;
        forgetting = ((0.90-0.65) + (0.60-0.55))/2 = 0.15 -- note the peak for
        task 0 is its own diagonal 0.90, not the post-CIFAR value 0.70, which is
        exactly the confusion that makes a hand-computed forgetting wrong;
        A_avg = (0.65+0.55+0.50)/3 = 0.5667."""
        bwt, forg, a_avg = _trainer_cl_metrics(self.MATRICES[0])
        self.assertAlmostEqual(bwt, -0.15, places=9)
        self.assertAlmostEqual(forg, 0.15, places=9)
        self.assertAlmostEqual(a_avg, 0.5666667, places=6)

    def test_a_avg_is_the_final_row_not_the_diagonal(self):
        """A_avg must average the last row. Averaging the diagonal would report
        only single-task accuracy and hide all forgetting."""
        M = self.MATRICES[0]
        _, _, a_avg = _trainer_cl_metrics(M)
        diag = (M[0][0] + M[1][1] + M[2][2]) / 3
        self.assertNotAlmostEqual(a_avg, diag, places=3)
        self.assertLess(a_avg, diag)

    def test_sign_of_bwt_is_not_absoluted(self):
        """Positive transfer must survive. The third matrix forgets nothing and
        improves on every task, so BWT > 0 and forgetting goes NEGATIVE (the
        final value is above the pre-final peak). Clipping either at zero would
        hide the difference between a plastic model and a stable one."""
        bwt, forg, _ = _trainer_cl_metrics(self.MATRICES[2])
        self.assertGreater(bwt, 0.0)
        self.assertLess(forg, 0.0, "negative forgetting = backward transfer")

    def test_zero_forgetting_snapshot_protocol(self):
        """A snapshot matrix has BWT = 0 and forgetting = 0 by construction; that
        is the property the README claims for A_oracle, so the metric must be
        able to express it exactly."""
        M = [[0.90, None, None], [0.90, 0.60, None], [0.90, 0.60, 0.50]]
        bwt, forg, a_avg = _trainer_cl_metrics(M)
        self.assertAlmostEqual(bwt, 0.0, places=9)
        self.assertAlmostEqual(forg, 0.0, places=9)
        self.assertAlmostEqual(a_avg, (0.90 + 0.60 + 0.50) / 3, places=9)


if __name__ == "__main__":
    unittest.main()