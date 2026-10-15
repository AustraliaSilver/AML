"""The in-place snapshot swap must be indistinguishable from load_state_dict.

`evaluate` swaps the encoder, selector, stem, 18 layers and the head LayerNorm
once per snapshot per batch. `load_state_dict` re-validates every key and shape
on each of those calls, which was measured to dominate evaluation (4.4 s).
`apply_state` copies the tensors straight into the live storage instead.

The speed-up is only legitimate if it is *invisible*, so these tests pin the
two properties that a hand-rolled copy could plausibly break:

  1. it writes into the live parameters rather than rebinding them, and
     produces exactly the values `load_state_dict` would have produced;
  2. after `evaluate` returns, the model's weights are bit-identical to what
     they were on entry - the swap is fully restored.

Property 2 is the one that matters for the continual protocol: a snapshot left
loaded would silently turn the "live weights" protocol into another oracle run.
"""
import unittest

import torch
import torch.nn as nn

from train_continual_aml import apply_state, evaluate, state_targets


class TestApplyState(unittest.TestCase):

    def _model(self):
        torch.manual_seed(0)
        return nn.Sequential(nn.Conv2d(3, 4, 3), nn.InstanceNorm2d(4),
                             nn.ReLU(), nn.Linear(4, 2))

    def test_matches_load_state_dict_value_for_value(self):
        m = self._model()
        src = {k: torch.randn_like(v) for k, v in m.state_dict().items()}
        expected = {k: v.clone() for k, v in m.state_dict().items()}
        for k, v in src.items():
            expected[k] = v.clone()

        apply_state(state_targets(m), src)

        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(v, expected[k]), f"{k} differs")

    def test_writes_into_live_tensors_instead_of_rebinding(self):
        """load_state_dict copies into the existing storage too. If apply_state
        ever replaced `p.data` instead, every optimizer that captured the
        parameter earlier would silently train a detached tensor.

        `state_dict()` hands back a *fresh* detached wrapper on every call, so
        object identity of those wrappers proves nothing; the parameter object
        and its storage address are what must stay put."""
        m = self._model()
        params_before = list(m.parameters())
        ptr_before = [p.data_ptr() for p in params_before]
        src = {k: torch.ones_like(v) for k, v in m.state_dict().items()}
        apply_state(state_targets(m), src)
        params_after = list(m.parameters())
        self.assertEqual([id(p) for p in params_before],
                         [id(p) for p in params_after])
        self.assertEqual(ptr_before, [p.data_ptr() for p in params_after])
        for p in params_after:
            self.assertTrue(torch.allclose(p, torch.ones_like(p)))

    def test_handles_dtype_and_device_mismatch(self):
        """Snapshots may be stored on CPU or int8-quantised (deq_protos). The
        copy must still land in the target's dtype/device."""
        m = self._model()
        tgt = state_targets(m)
        src = {k: torch.ones_like(v).double().cpu() for k, v in m.state_dict().items()}
        apply_state(tgt, src)
        for k, v in m.state_dict().items():
            self.assertEqual(v.dtype, torch.float32)
            self.assertTrue(torch.allclose(v, torch.ones_like(v)))


class TestEvaluateRestoresLiveWeights(unittest.TestCase):

    def _setup(self):
        from mgn.blocks import ConvAML_18Layer_Hybrid
        torch.manual_seed(1)
        m = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=8, num_classes=10,
                                   norm_type="in", encoder_tail_act="none").eval()
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        loader = [(x, y)]

        # snapshot 0 is a DIFFERENT function from the live weights, so a swap
        # that is skipped or not restored is detectable rather than a no-op
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        snaps = [{
            "enc": {k: v.detach().clone() for k, v in m.vision_encoder.state_dict().items()},
            "sel": {k: v.detach().clone() for k, v in m.prototype_selector.state_dict().items()},
            "stem": {k: v.detach().clone() for k, v in m.stem.state_dict().items()},
            "layers": {k: v.detach().clone() for k, v in m.layers.state_dict().items()},
            "ln": {k: v.detach().clone() for k, v in m.ln.state_dict().items()},
            "struct_heads": {"sprotos": m.structure_prototypes.detach().clone(),
                             "bmprotos": m.block_mode_prototypes.detach().clone()},
            "protos": m.proto_weights.detach().clone(),
        }]
        # now move the live weights away from the snapshot
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(0.5)
        with torch.no_grad():
            _, st, _, _ = m(x, tau=0.1)
        slots = [{"gate": torch.ones(m.max_layers),
                  "bm_mode": st.block_mode[0].clone(),
                  "bm_res": st.block_mode[0, :, 1].clone()}]

        class _Head(nn.Module):
            def __init__(self, n=1):
                super().__init__()
                self.l = nn.Linear(m.latent_dim, n)

            def forward(self, z):
                return self.l(z)

        return m, loader, snaps, slots, _Head

    def test_weights_are_bit_identical_after_evaluate(self):
        m, loader, snaps, slots, head = self._setup()
        before = {k: v.detach().clone() for k, v in m.state_dict().items()}
        before_extra = {
            "proto_weights": m.proto_weights.detach().clone(),
            "structure_prototypes": m.structure_prototypes.detach().clone(),
            "block_mode_prototypes": m.block_mode_prototypes.detach().clone(),
        }
        evaluate(m, loader, torch.device("cpu"), dom_head=head(), slots=slots,
                 snaps=snaps, route_tau=0.6, dom_truth=0)
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), f"{k} was not restored")
        self.assertTrue(torch.equal(m.proto_weights, before_extra["proto_weights"]))
        self.assertTrue(torch.equal(m.structure_prototypes,
                                    before_extra["structure_prototypes"]))
        self.assertTrue(torch.equal(m.block_mode_prototypes,
                                    before_extra["block_mode_prototypes"]))

    def test_two_calls_give_the_same_accuracy(self):
        """If the swap leaked state between batches or between calls, the second
        pass would be scored under different weights than the first."""
        m, loader, snaps, slots, head = self._setup()
        a1, _ = evaluate(m, loader, torch.device("cpu"), dom_head=head(), slots=slots,
                         snaps=snaps, route_tau=0.6, dom_truth=0)
        a2, _ = evaluate(m, loader, torch.device("cpu"), dom_head=head(), slots=slots,
                         snaps=snaps, route_tau=0.6, dom_truth=0)
        self.assertEqual(a1, a2)

    def test_snapshot_actually_changes_the_prediction(self):
        """Guards the tests above from passing vacuously: if the snapshot were
        identical to the live weights, 'restored' would be indistinguishable
        from 'never swapped'."""
        from mgn.blocks import ConvAML_18Layer_Hybrid
        m, loader, snaps, _, _ = self._setup()

        snap_model = ConvAML_18Layer_Hybrid(
            K=4, max_layers=18, max_dim=8, num_classes=10,
            norm_type="in", encoder_tail_act="none").eval()
        for mod, key in ((snap_model.vision_encoder, "enc"),
                         (snap_model.prototype_selector, "sel"),
                         (snap_model.stem, "stem"),
                         (snap_model.layers, "layers"),
                         (snap_model.ln, "ln")):
            mod.load_state_dict(snaps[0][key])
        with torch.no_grad():
            snap_model.structure_prototypes.copy_(snaps[0]["struct_heads"]["sprotos"])
            snap_model.block_mode_prototypes.copy_(snaps[0]["struct_heads"]["bmprotos"])
            snap_model.proto_weights.copy_(snaps[0]["protos"])
            o_live, _, _, _ = m(loader[0][0], tau=0.1)
            o_snap, _, _, _ = snap_model(loader[0][0], tau=0.1)
        self.assertFalse(torch.equal(o_live, o_snap),
                         "the fixture must actually differ from the snapshot")


if __name__ == "__main__":
    unittest.main()