"""
Script to count exact parameters of Child Model and HtS Hypernetwork
"""
import torch
from hts_streaming_sota import HtSSOTAModel

def count_parameters():
    model = HtSSOTAModel(d=64, num_layers=8, num_classes=10)
    
    stem_mnist_p = sum(p.numel() for p in model.stem_mnist.parameters())
    stem_cifar_p = sum(p.numel() for p in model.stem_cifar.parameters())
    
    layer_params = []
    for i, layer in enumerate(model.child_layers):
        p_count = sum(p.numel() for p in layer.parameters())
        layer_params.append(p_count)
    
    head_mnist_p = sum(p.numel() for p in model.head_mnist.parameters())
    head_cifar_p = sum(p.numel() for p in model.head_cifar.parameters())
    
    child_mnist_active = stem_mnist_p + sum(layer_params) + head_mnist_p
    child_cifar_active = stem_cifar_p + sum(layer_params) + head_cifar_p
    
    hts_basis_p = model.hts.hts_basis.numel()
    shared_proto_p = model.hts.shared_prototypes.numel()
    task_proj_p = sum(p.numel() for p in model.hts.task_proj.parameters())
    router_p = sum(p.numel() for p in model.hts.router.parameters())
    coeff_gen_p = sum(p.numel() for p in model.hts.coeff_gen.parameters())
    task_encoder_p = sum(p.numel() for p in model.task_state_encoder.parameters())
    
    total_hypernet = hts_basis_p + shared_proto_p + task_proj_p + router_p + coeff_gen_p + task_encoder_p
    total_all = sum(p.numel() for p in model.parameters())
    
    print("="*65)
    print("THONG KE THAM SO CHI TIET (MANG CON & HtS SYSTEM)")
    print("="*65)
    print(f"1. MANG CON (CHILD MODEL INFERENCE - D=64, 8 LAYERS):")
    print(f"   - Stem Native (MNIST)        : {stem_mnist_p:>10,}")
    print(f"   - Stem Native (CIFAR-10)     : {stem_cifar_p:>10,}")
    print(f"   - 8 Dynamic Child Layers     : {sum(layer_params):>10,} ({sum(layer_params)//8:,}/layer)")
    print(f"     * Conv mode path           : {sum(p.numel() for l in model.child_layers for p in l.conv.parameters()):>10,}")
    print(f"     * MLP mode path            : {sum(p.numel() for l in model.child_layers for p in l.mlp.parameters()):>10,}")
    print(f"     * ResNet mode path         : {sum(p.numel() for l in model.child_layers for p in [l.res_conv1, l.res_conv2] for p in p.parameters()):>10,}")
    print(f"     * ViT-like mode path       : {sum(p.numel() for l in model.child_layers for p in [l.qkv, l.proj] for p in p.parameters()):>10,}")
    print(f"   - Classification Head (MNIST): {head_mnist_p:>10,}")
    print(f"   ----------------------------------------------------------")
    print(f"   * TONG ACTIVE PARAMS MANG CON (MNIST) : {child_mnist_active:>10,}")
    print(f"   * TONG ACTIVE PARAMS MANG CON (CIFAR) : {child_cifar_active:>10,}")
    print(f"\n2. HtS HYPERNETWORK KNOWLEDGE SYSTEM:")
    print(f"   - Shared HtS Basis Tensor    : {hts_basis_p:>10,}")
    print(f"   - Shared Prototypes Bank     : {shared_proto_p:>10,}")
    print(f"   - Task Controller Projections: {task_proj_p:>10,}")
    print(f"   - Dynamic Mode Router        : {router_p:>10,}")
    print(f"   - Basis Coeff Generator      : {coeff_gen_p:>10,}")
    print(f"   - Task State Estimator       : {task_encoder_p:>10,}")
    print(f"   ----------------------------------------------------------")
    print(f"   * TONG HtS HYPERNETWORK PARAMS       : {total_hypernet:>10,}")
    print(f"\n* TONG TOAN BO HE THONG             : {total_all:>10,}")
    print("="*65)

if __name__ == '__main__':
    count_parameters()
