import unittest
import torch
from mgn.generator import ModelGenerator
from mgn.executor import TaskNetwork
from mgn.constraints import compute_aml_efficiency, compute_aml_efficiency_torch
from hts_lib.models.hypernetwork import HtSContinualModel
from hts_lib.data.memory import EpisodicMemoryBuffer
from hts_lib.evaluation.evaluator import ContinualEvaluator

class TestAMLUnified(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_mgn_generator_and_task_network(self):
        """Test AML ModelGenerator and TaskNetwork forward pass."""
        gen = ModelGenerator(
            input_dim=16, output_dim=4, max_layers=3, max_dim=16,
            num_prototypes=4, use_conv=False
        ).to(self.device)
        executor = TaskNetwork().to(self.device)
        
        x = torch.randn(4, 16, device=self.device)
        model = gen(x)
        out = executor(x, model)
        
        self.assertEqual(out.shape, (4, 4))
        self.assertEqual(model.structure.layer_gates.shape, (4, 3))

    def test_c_infinity_efficiency_metric(self):
        """Test smooth C^infinity efficiency metric calculation."""
        # Standard numpy efficiency
        e_np = compute_aml_efficiency(accuracy=0.85, p_allowance=100000, q_used=50000)
        self.assertGreater(e_np, 0.0)
        
        # Differentiable PyTorch efficiency
        acc_t = torch.tensor(0.85, device=self.device)
        q_t = torch.tensor(50000.0, device=self.device)
        e_torch = compute_aml_efficiency_torch(acc_t, p_allowance=100000, q_used=q_t)
        self.assertGreater(e_torch.item(), 0.0)

    def test_hts_continual_hypernetwork(self):
        """Test HtS Dual-Residual Dynamic Weight Synthesis."""
        hts = HtSContinualModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(self.device)
        x_cifar = torch.randn(2, 3, 32, 32, device=self.device)
        out, alphas = hts(x_cifar, task="cifar")
        
        self.assertEqual(out.shape, (2, 10))
        self.assertEqual(alphas.shape, (2, 8, 8))

    def test_episodic_memory(self):
        """Test Episodic Memory Buffer under strict constraints."""
        mem = EpisodicMemoryBuffer(max_total_samples=100)
        dummy_dataset = [(torch.randn(3, 32, 32), i % 10) for i in range(200)]
        mem.update_with_dataset("task1", dummy_dataset)
        self.assertEqual(mem.get_total_stored_samples(), 100)

if __name__ == '__main__':
    unittest.main()
