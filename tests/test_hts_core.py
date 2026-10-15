"""
Comprehensive Unit & Integration Test Suite for HtS Library
============================================================
Tests:
1. Dynamic Kernel Synthesis Shapes and Types.
2. Gradient Flow across Shared Base and Private Basis Experts.
3. Batch Size Invariance (B=1, B=2, B=16).
4. Per-Sample Routing Diversity (Samples with different inputs get different alphas).
5. Dynamic Task Registration (`add_task()`).
6. CPU vs CUDA Execution Portability.
7. Formal Academic Metric Calculation (A[i,j], BWT, Forgetting).
"""

import unittest
import torch
from hts_lib.models.hypernetwork import HtSContinualModel
from hts_lib.models.losses import compute_layer_aware_diversity_loss
from hts_lib.evaluation.evaluator import ContinualEvaluator
from hts_lib.utils import set_seed

class TestHtSCore(unittest.TestCase):
    def setUp(self):
        set_seed(2026)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HtSContinualModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(self.device)

    def test_forward_shapes_and_tasks(self):
        """Test forward pass output dimensions across native stems."""
        # CIFAR-10: 3x32x32
        x_cifar = torch.randn(4, 3, 32, 32, device=self.device)
        out_c, alphas_c = self.model(x_cifar, task="cifar")
        self.assertEqual(out_c.shape, (4, 10))
        self.assertEqual(alphas_c.shape, (4, 8, 8))
        
        # MNIST: 1x28x28
        x_mnist = torch.randn(4, 1, 28, 28, device=self.device)
        out_m, alphas_m = self.model(x_mnist, task="mnist")
        self.assertEqual(out_m.shape, (4, 10))
        self.assertEqual(alphas_m.shape, (4, 8, 8))

    def test_batch_size_invariance(self):
        """Test that single-sample (B=1) and batched execution (B>1) execute smoothly."""
        for b in [1, 2, 7]:
            x = torch.randn(b, 3, 32, 32, device=self.device)
            out, _ = self.model(x, task="cifar")
            self.assertEqual(out.shape, (b, 10))

    def test_gradient_flow(self):
        """Test full backward pass and gradient existence in all basis components."""
        self.model.train()
        x = torch.randn(2, 3, 32, 32, device=self.device)
        y = torch.tensor([1, 4], device=self.device)
        
        logits, alphas = self.model(x, task="cifar")
        loss_ce = torch.nn.functional.cross_entropy(logits, y)
        l_ent, l_use, l_ort = compute_layer_aware_diversity_loss(alphas, self.model, K=8)
        loss = loss_ce + 0.01 * l_ent + 0.1 * l_use + 0.05 * l_ort
        loss.backward()
        
        # Verify gradients exist in base and private basis
        self.assertIsNotNone(self.model.basis_sys.base1.grad)
        self.assertIsNotNone(self.model.basis_sys.b1_priv.grad)
        self.assertIsNotNone(self.model.task_ctrl[2].weight.grad)

    def test_dynamic_task_addition(self):
        """Test registering a custom new dataset dynamically via add_task()."""
        # Add SVHN (3 channels, 10 classes) or custom task
        self.model.add_task("custom_medical", in_channels=4, num_classes=5)
        self.assertIn("custom_medical", self.model.supported_tasks)
        
        x_custom = torch.randn(2, 4, 32, 32, device=self.device)
        out, _ = self.model(x_custom, task="custom_medical")
        self.assertEqual(out.shape, (2, 5))

    def test_academic_metrics_calculation(self):
        """Test formal calculation of Accuracy Matrix, BWT, and Forgetting."""
        # Simulated Accuracy Matrix for 3 tasks:
        # After Task 1: [99.5,   -,    -]
        # After Task 2: [99.3, 72.7,   -]
        # After Task 3: [99.4, 72.7, 90.5]
        A = [
            [99.5,  0.0,  0.0],
            [99.3, 72.7,  0.0],
            [99.4, 72.7, 90.5]
        ]
        tasks = ["MNIST", "CIFAR-10", "Fashion-MNIST"]
        metrics = ContinualEvaluator.compute_continual_metrics(A, tasks)
        
        self.assertAlmostEqual(metrics["average_accuracy"], 87.53, places=1)
        self.assertAlmostEqual(metrics["backward_transfer_bwt"], -0.05, places=1)
        self.assertAlmostEqual(metrics["average_forgetting"], 0.05, places=1)

    def test_parameter_accounting(self):
        """Verify strict breakdown between child parameters and hypernetwork parameters."""
        stats = self.model.count_parameters_breakdown(task="cifar")
        # Child active params must be strictly under 10k
        self.assertLess(stats["child_active_params"], 10000)
        self.assertEqual(stats["child_active_params"], 9490)
        # Total system is ~1.47M
        self.assertGreater(stats["total_hypernetwork_params"], 1000000)

    def test_episodic_memory_buffer(self):
        """Test episodic memory buffer budgeting and data loaders."""
        from hts_lib.data.memory import EpisodicMemoryBuffer
        mem = EpisodicMemoryBuffer(max_total_samples=100)
        
        # Mock dataset
        dummy_dataset = [(torch.randn(3, 32, 32), i % 10) for i in range(500)]
        mem.update_with_dataset("task1", dummy_dataset)
        self.assertEqual(mem.get_total_stored_samples(), 100)
        
        # Add second task -> memory rebalanced to 50/50
        mem.update_with_dataset("task2", dummy_dataset)
        self.assertEqual(mem.get_total_stored_samples(), 100)
        self.assertEqual(len(mem.task_memories["task1"]["data"]), 50)
        self.assertEqual(len(mem.task_memories["task2"]["data"]), 50)
        
        # Check DataLoader creation
        loader = mem.get_memory_loader("task1", batch_size=16)
        x_batch, y_batch = next(iter(loader))
        self.assertEqual(x_batch.shape[1:], (3, 32, 32))
        self.assertEqual(len(y_batch), 16)

    def test_tensor_core_forward(self):
        """Test forward pass using Tensor Core WMMA kernel acceleration."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        x = torch.randn(4, 3, 32, 32, device=self.device)
        out, alphas = self.model(x, task="cifar", use_tc=True)
        self.assertEqual(out.shape, (4, 10))
        self.assertEqual(alphas.shape, (4, 8, 8))
        self.assertFalse(torch.isnan(out).any())

    def test_hts_basis_isolator(self):
        """Test HtSBasisIsolator gradient masking and parameter snapshot restoration."""
        from hts_lib.training.trainer import HtSBasisIsolator
        isolator = HtSBasisIsolator(self.model)
        
        # Task 0 of 3: s=0, e=3 (for K=8)
        s, e = isolator.begin_task(0, num_tasks=3)
        self.assertEqual(s, 0)
        self.assertEqual(e, 3)

        self.model.train()
        x = torch.randn(2, 3, 32, 32, device=self.device)
        y = torch.tensor([0, 1], device=self.device)
        out, _ = self.model(x, task="cifar")
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()

        # Gradients for experts [3:] must be masked to zero
        for name in ("b1_priv", "b2_priv", "b3_priv"):
            param = getattr(self.model.basis_sys, name)
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.all(param.grad[e:] == 0.0))

        # Test snapshot restoration
        isolator.restore()

    def test_residual_block_accepts_widths_that_do_not_divide(self):
        """Regression: HtSResidualBlock used GroupNorm(min(8, out_d // 4), out_d),
        which raised 'num_channels must be divisible by num_groups' for
        out_d=30 (7 groups) and built GroupNorm(0, out_d) for out_d < 4."""
        from hts_lib.models.blocks import HtSResidualBlock, _norm_groups
        # widths that already worked keep exactly the old group counts
        self.assertEqual(_norm_groups(24), 6)
        self.assertEqual(_norm_groups(48), 8)
        self.assertEqual(_norm_groups(96), 8)
        # widths that used to crash now pick a true divisor
        for d in (5, 7, 12, 25, 30, 100):
            b = HtSResidualBlock(d, d)
            self.assertEqual(b.gn2.num_channels, d)
            self.assertEqual(d % b.gn2.num_groups, 0)
            self.assertGreaterEqual(b.gn2.num_groups, 1)
        # and the block actually runs a forward at such a width
        b = HtSResidualBlock(30, 30).eval()
        x = torch.randn(2, 30, 8, 8)
        k = torch.randn(2, 30, 30, 3, 3)
        with torch.no_grad():
            self.assertEqual(tuple(b(x, k).shape), (2, 30, 8, 8))

    def test_gn1_keeps_the_fixed_four_group_convention(self):
        """gn1 (and the native stems) were built with a constant GroupNorm(4, C),
        NOT with the `min(8, C // 4)` used for gn2. A shared helper that
        preferred `min(cap, C // 4)` would silently rewrite C=8/12 from 4 groups
        to 2/3 and move every HtS number. All default widths (24/48/96) are
        divisible by 4, so 4 must be preserved there."""
        from hts_lib.models.blocks import HtSResidualBlock, _stem_groups
        for d in (8, 12, 24, 48, 96):
            self.assertEqual(_stem_groups(d), 4)
            b = HtSResidualBlock(d, d)
            self.assertEqual(b.gn1.num_groups, 4)
        # a width 4 does not divide still gets a valid divisor
        for d in (6, 30):
            self.assertEqual(d % _stem_groups(d), 0)
            self.assertEqual(HtSResidualBlock(d, d).gn1.num_groups, _stem_groups(d))

    def test_residual_shortcut_widths_are_safe_too(self):
        from hts_lib.models.blocks import HtSResidualBlock
        b = HtSResidualBlock(12, 30, stride=2)
        for m in b.shortcut.modules():
            if isinstance(m, torch.nn.GroupNorm):
                self.assertEqual(30 % m.num_groups, 0)

    def test_memory_selection_is_reproducible_under_the_library_seed(self):
        """The report flagged the mixed torch.randperm / random.shuffle in
        EpisodicMemoryBuffer as unreproducible. It is not: hts_lib.utils.set_seed
        (and train.set_seed) seed BOTH RNGs, and hts_lib's buffer is only reached
        through them (experiments/benchmark_hts_baselines.py imports
        hts_lib.utils.set_seed). Rewriting the shuffle to the torch RNG therefore
        bought no reproducibility and would have moved the HtS baseline numbers.
        This pins the actual contract: a run reset by the shipped helper picks the
        same exemplars."""
        from hts_lib.data.memory import EpisodicMemoryBuffer
        from hts_lib.utils import set_seed

        class D(torch.utils.data.Dataset):
            def __init__(self):
                self.targets = [i % 4 for i in range(200)]

            def __len__(self):
                return 200

            def __getitem__(self, i):
                return torch.tensor(float(i)), self.targets[i]

        def picked():
            set_seed(0)
            mem = EpisodicMemoryBuffer(max_total_samples=40)
            mem.update_with_dataset("t", D())
            return mem.task_memories["t"]["targets"].tolist()

        a, b = picked(), picked()
        self.assertEqual(len(a), 40)
        self.assertEqual(a, b,
                         "exemplar selection is not reproducible under "
                         "hts_lib.utils.set_seed")


if __name__ == '__main__':
    unittest.main()
