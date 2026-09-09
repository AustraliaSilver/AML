import torch
import mgn.accel._C as _C
import time

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def run_benchmark():
    B = 128
    # True sizes based on architecture: NO pooling in TaskNetwork mid-layers!
    Hs = [32]*15
    Ws = [32]*15
    C_ins = [48]*15
    C_outs = [48]*15
    max_dim = 32

    X = torch.randn(B, 48, 32, 32, device='cuda')
    pi = torch.randn(B, 4, device='cuda')
    prototypes = torch.randn(4, 48*48*9*15 + 48*15, device='cuda')
    layer_gates = torch.ones(B, 15, device='cuda')
    dim_masks = torch.ones(B, 15, 32*32, device='cuda')
    block_mode = torch.zeros(B, 15, 2, device='cuda')

    w_offsets = [i*48*48*9 for i in range(15)]
    b_offsets = [15*48*48*9 + i*48 for i in range(15)]

    def forward(X_, pi_):
        return _C.fused_forward(X_, prototypes, pi_, layer_gates, dim_masks, block_mode, w_offsets, b_offsets, C_ins, C_outs, Hs, Ws, max_dim)

    print("================================================================")
    print("    Option 1: CUDAGraphs Pipeline Fusion Benchmark (15 Layers)  ")
    print("================================================================")
    
    # Warmup
    print('USING _C:', _C.__file__)
    for _ in range(5):
        forward(X, pi)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        forward(X, pi)
    torch.cuda.synchronize()
    t1 = time.time()
    no_graph_time = (t1-t0)/10*1000
    print(f'-> [Before] Pipeline C++ (No Graph, FP32): {no_graph_time:.2f} ms')

    # Build CUDA Graph
    graphed = torch.cuda.make_graphed_callables(forward, (X, pi))

    for _ in range(5):
        graphed(X, pi)
    torch.cuda.synchronize()

    t2 = time.time()
    for _ in range(10):
        graphed(X, pi)
    torch.cuda.synchronize()
    t3 = time.time()
    graph_time = (t3-t2)/10*1000
    print(f'-> [After] Pipeline C++ (CUDAGraphs Fusion): {graph_time:.2f} ms')
    
    speedup = no_graph_time / graph_time
    print(f'-> Speedup: {speedup:.2f}x ({(speedup - 1)*100:.1f}% faster)')
    print("================================================================")

if __name__ == '__main__':
    run_benchmark()
