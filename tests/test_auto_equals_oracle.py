"""A perfect router must reproduce the oracle exactly.

The whole point of the router work is that A_auto should approach A_oracle. It
got there in stages (A[2,2]: 8.01 -> 22.00 -> 16.37 -> 13.05 -> 9.67 -> 33.55,
i.e. exactly the oracle) and each stage exposed a different stacked cause. None
of those stages has a test, so any of them can silently come back: the router
may start predicting correctly again while something *downstream* of it drifts.

This test closes that gap. It stands in a router that is correct by construction
and asserts the two protocols are numerically identical, sample for sample. A
regression anywhere between `d_pred` and the prediction - the per-sample
prototype mask, the structure-slot override, the snapshot swap, the ambiguous
retry - moves the number, so the test fails with a diff rather than silently
degrading the headline.

The router is a constant-logit head rather than a learned one on purpose: the
assertion is about what the *rest* of the path does with a correct decision, and
a learned router would let a wrong-but-confident prediction hide a downstream
regression (or fail for reasons unrelated to what is being tested).
`route_acc` is asserted to be 1.0 first, so the test cannot pass vacuously.
"""
import unittest

import torch
import torch.nn as nn

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid
from train_continual_aml import evaluate


class _ConstHead(nn.Module):
    """A router whose answer never depends on the features."""

    def __init__(self, dim, n_tasks):
        super().__init__()
        self.lin = nn.Linear(dim, n_tasks)

    def set_answer(self, t):
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.fill_(-50.0)
            self.lin.bias[t] = 50.0

    def forward(self, z):
        return self.lin(z)


class TestAutoMatchesOracle(unittest.TestCase):

    T = 2
    K = 6
    B = 4

    def _fixture(self, seed=0):
        torch.manual_seed(seed)
        m = ConvAML_18Layer_Hybrid(K=self.K, max_layers=18, max_dim=8,
                                   num_classes=10, norm_type="in",
                                   encoder_tail_act="none").eval()
        xs = [torch.randn(self.B, 3, 32, 32) for _ in range(self.T)]

        def snap():
            return {
                "enc": {k: v.detach().clone() for k, v in m.vision_encoder.state_dict().items()},
                "sel": {k: v.detach().clone() for k, v in m.prototype_selector.state_dict().items()},
                "stem": {k: v.detach().clone() for k, v in m.stem.state_dict().items()},
                "layers": {k: v.detach().clone() for k, v in m.layers.state_dict().items()},
                "ln": {k: v.detach().clone() for k, v in m.ln.state_dict().items()},
                "struct_heads": {
                    "sprotos": m.structure_prototypes.detach().clone(),
                    "bmprotos": m.block_mode_prototypes.detach().clone(),
                },
                "protos": m.proto_weights.detach().clone(),
            }

        snaps = [snap()]
        # A genuinely different second domain: different trunk weights AND
        # different structure/prototype parameters, so a mis-swap cannot go
        # unnoticed.
        with torch.no_grad():
            for mod in (m.vision_encoder, m.prototype_selector, m.stem, m.layers, m.ln):
                for p in mod.parameters():
                    p.add_(torch.randn_like(p) * 0.5)
            m.structure_prototypes.add_(torch.randn_like(m.structure_prototypes) * 0.5)
            m.block_mode_prototypes.add_(torch.randn_like(m.block_mode_prototypes) * 0.5)
        snaps.append(snap())

        def load_snap(t):
            for mod, key in ((m.vision_encoder, "enc"),
                             (m.prototype_selector, "sel"),
                             (m.stem, "stem"),
                             (m.layers, "layers"),
                             (m.ln, "ln")):
                mod.load_state_dict(snaps[t][key])
            with torch.no_grad():
                m.structure_prototypes.copy_(snaps[t]["struct_heads"]["sprotos"])
                m.block_mode_prototypes.copy_(snaps[t]["struct_heads"]["bmprotos"])
                m.proto_weights.copy_(snaps[t]["protos"])

        # Labels are what each domain ACTUALLY predicts under its own snapshot.
        # With random labels both protocols score 0.0 and "auto == oracle"
        # becomes a vacuous identity between two zeros.
        ys, slots = [], []
        for t in range(self.T):
            load_snap(t)
            with torch.no_grad():
                out, st, _, _ = m(xs[t], tau=0.1)
            ys.append(out.argmax(-1))
            slots.append({"gate": torch.ones(m.max_layers),
                          "bm_mode": st.block_mode[0].clone(),
                          "bm_res": st.block_mode[0, :, 1].clone()})
        loaders = [[(xs[t], ys[t])] for t in range(self.T)]

        # Live weights move again after the last snapshot was taken.
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(0.9)

        ranges = [(round(k * self.K / self.T), round((k + 1) * self.K / self.T))
                  for k in range(self.T)]
        head = _ConstHead(m.latent_dim, self.T)
        return m, loaders, snaps, slots, head, ranges

    def _pi_mask(self, t):
        pm = torch.zeros(self.K, dtype=torch.bool)
        pm[round(t * self.K / self.T):round((t + 1) * self.K / self.T)] = True
        return pm

    def test_auto_equals_oracle_when_routing_is_perfect(self):
        m, loaders, snaps, slots, head, ranges = self._fixture()
        dev = torch.device("cpu")
        for t in range(self.T):
            head.set_answer(t)

            acc_oracle, _ = evaluate(
                m, loaders[t], dev, dom_head=head, slots=slots, snaps=snaps,
                route_tau=0.6, dom_truth=t, pi_mask=self._pi_mask(t))

            acc_auto, fp_auto = evaluate(
                m, loaders[t], dev, dom_head=head, slots=slots, snaps=snaps,
                route_tau=0.6, dom_truth=None, pi_mask=None, route_truth=t,
                proto_row_ranges=ranges)

            self.assertEqual(fp_auto.get("route_acc"), 1.0,
                             f"the stub router must be right on task {t}")
            self.assertEqual(acc_oracle, 1.0,
                             "the fixture must be solvable, or equality below "
                             "would just be 0.0 == 0.0")
            self.assertEqual(acc_auto, acc_oracle,
                             f"task {t}: autonomous {acc_auto:.4f} != "
                             f"oracle {acc_oracle:.4f} with a perfect router")

    def test_snapshots_must_actually_differ(self):
        """Guards the test above from passing vacuously: if the two snapshots
        computed the same function, 'equal to oracle' would hold for any
        downstream code, correct or not."""
        m, loaders, snaps, slots, head, ranges = self._fixture()
        ys = [loaders[t][0][1] for t in range(self.T)]
        self.assertFalse(torch.equal(ys[0], ys[1]),
                         "the two snapshots must predict different labels, "
                         "otherwise a mis-swap is undetectable")

    def test_wrong_routing_does_change_the_answer(self):
        """The converse: the autonomous path is not accidentally constant. If a
        deliberately wrong router still scored the same, the test above would be
        measuring nothing."""
        m, loaders, snaps, slots, head, ranges = self._fixture()
        dev = torch.device("cpu")
        t = 0
        head.set_answer(0)
        acc_right, fp = evaluate(
            m, loaders[t], dev, dom_head=head, slots=slots, snaps=snaps,
            route_tau=0.6, dom_truth=None, pi_mask=None, route_truth=t,
            proto_row_ranges=ranges)
        head.set_answer(1)                     # route everything to domain 1
        acc_wrong, _ = evaluate(
            m, loaders[t], dev, dom_head=head, slots=slots, snaps=snaps,
            route_tau=0.6, dom_truth=None, pi_mask=None, route_truth=t,
            proto_row_ranges=ranges)
        self.assertEqual(fp.get("route_acc"), 1.0)
        self.assertNotEqual(acc_right, acc_wrong,
                            "routing must actually influence the result")


    def test_ambiguous_retry_uses_the_retried_domain_s_slot(self):
        """A sample retried under snapshot d must use d's STRUCTURE, not d's rival's.

        The ambiguous path loads snapshot d's weights, selects d's prototype
        rows... and used to take the layer gates and block modes from
        `d_pred`, the router's argmax. That evaluates domain d's weights under
        another domain's architecture. `slots[d]` is collected from d's own data
        with d's own weights (collect_structure_slot), so it describes d and
        nothing else.

        This is not a corner case: `route_tau` defaults to 0.6 and `amb_topk` to
        0, so every seen domain is retried for every ambiguous sample on every
        default run. Nothing tested it, because the fixture's router is a
        +/-50-logit head whose margin is ~1.0, i.e. never ambiguous at 0.6. The
        test therefore forces the branch with `route_tau=1.1`, which no softmax
        margin can exceed.

        The two slots are made distinguishable (closed vs open gates) and the
        gates actually handed to the model are recorded, so the assertion is on
        what was executed rather than on what the code reads like.
        """
        m, loaders, snaps, slots, head, ranges = self._fixture()
        dev = torch.device("cpu")
        L = m.max_layers
        slots[0]["gate"] = torch.zeros(L)   # domain 0: every layer closed
        slots[1]["gate"] = torch.ones(L)    # domain 1: every layer open

        seen = []
        real_forward = m.forward

        def spy(*a, **kw):
            so = kw.get("struct_override", None)
            seen.append(None if so is None else so[0].detach().clone())
            return real_forward(*a, **kw)

        m.forward = spy
        try:
            head.set_answer(0)              # the router is certain it is domain 0
            evaluate(m, loaders[0], dev, dom_head=head, slots=slots, snaps=snaps,
                     route_tau=1.1, dom_truth=None, pi_mask=None, route_truth=0,
                     proto_row_ranges=ranges)
        finally:
            m.forward = real_forward

        self.assertEqual(len(seen), len(snaps),
                         "every seen domain must have been retried")
        self.assertFalse(torch.allclose(seen[0], seen[1]),
                         "both retries ran with the same gate slot: the slot came "
                         "from d_pred instead of from the domain being retried")
        for d, g in enumerate(seen):
            self.assertTrue(torch.allclose(g, slots[d]["gate"].expand_as(g)),
                            f"the retry under snapshot {d} used the wrong gate slot")


if __name__ == "__main__":
    unittest.main()