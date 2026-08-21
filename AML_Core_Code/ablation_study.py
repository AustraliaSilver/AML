import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
import collections
import os

DATA_PATH = r"C:\Users\hunga\minecraft_ai_dataset.jsonl"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define neural network class for dynamic architectures
class MinecraftAblationMLP(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size):
        super(MinecraftAblationMLP, self).__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# 2. Read dataset and extract single-frame raw features and actions
print("=== MINECRAFT AI ABLATION STUDY ===")
print(f"Device: {device}")
print(f"Reading dataset from: {DATA_PATH} ...")

target_map = {"MISS": 0, "BLOCK": 1, "WATER": 2, "LAVA": 3, "ENTITY": 4}
raw_features_list = []
raw_action_ids_list = []
unique_actions = []

if not os.path.exists(DATA_PATH):
    print(f"Error: Dataset file not found at {DATA_PATH}")
    exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        
        state = record.get("state", {})
        action = record.get("action", {})
        
        pos = state.get("pos", {})
        states = state.get("states", {})
        
        features = [
            float(state.get("hp", 20)),
            float(state.get("food", 20)),
            float(state.get("xp_lvl", 0)),
            float(pos.get("pitch", 0)),
            float(pos.get("yaw", 0)),
            1.0 if states.get("sprint") else 0.0,
            1.0 if states.get("sneak") else 0.0,
            1.0 if states.get("ground") else 0.0,
            float(target_map.get(state.get("target", "MISS"), 0))
        ]
        
        act_str = json.dumps(action, sort_keys=True)
        if act_str not in unique_actions:
            unique_actions.append(act_str)
        action_id = unique_actions.index(act_str)
        
        raw_features_list.append(features)
        raw_action_ids_list.append(action_id)

output_size = len(unique_actions)
total_raw_records = len(raw_features_list)
print(f"[+] Total raw records loaded: {total_raw_records}")
print(f"[+] Total unique actions: {output_size}\n")

# Helper function to construct windowed tensors for history length N
def get_windowed_tensors(N):
    X_raw = []
    Y_raw = []
    history_buffer = collections.deque(maxlen=N)
    
    for feat, act in zip(raw_features_list, raw_action_ids_list):
        history_buffer.append(feat)
        if len(history_buffer) == N:
            flattened = []
            for frame in history_buffer:
                flattened.extend(frame)
            X_raw.append(flattened)
            Y_raw.append(act)
            
    X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.long, device=device)
    return X_tensor, Y_tensor

# Helper function to train model and measure metrics
def run_training(X_tensor, Y_tensor, hidden_dims, epochs=100, lr=0.01, checkpoints=None):
    input_size = X_tensor.shape[1]
    num_samples = X_tensor.shape[0]
    
    # Fix seed for fair comparison across experiments
    torch.manual_seed(42)
    
    model = MinecraftAblationMLP(input_size, hidden_dims, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    checkpoint_results = {}
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        
        if checkpoints and epoch in checkpoints:
            elapsed = time.time() - start_time
            with torch.no_grad():
                model.eval()
                eval_outputs = model(X_tensor)
                eval_loss = criterion(eval_outputs, Y_tensor).item()
                _, predicted = torch.max(eval_outputs, 1)
                accuracy = (predicted == Y_tensor).sum().item() / num_samples * 100
                checkpoint_results[epoch] = {
                    "time": elapsed,
                    "loss": eval_loss,
                    "accuracy": accuracy
                }
                
    end_time = time.time()
    total_time = end_time - start_time
    
    with torch.no_grad():
        model.eval()
        final_outputs = model(X_tensor)
        final_loss = criterion(final_outputs, Y_tensor).item()
        _, predicted = torch.max(final_outputs, 1)
        final_accuracy = (predicted == Y_tensor).sum().item() / num_samples * 100
        
    return {
        "input_size": input_size,
        "num_samples": num_samples,
        "training_time": total_time,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "checkpoint_results": checkpoint_results
    }


# =========================================================================
# 1. HISTORY WINDOW SIZE STUDY
# =========================================================================
print("Running Ablation 1: History Window Sizes (N = 1, 5, 10, 20, 30)...")
window_sizes = [1, 5, 10, 20, 30]
deep_dims = [128, 256, 128]
window_results = []

for N in window_sizes:
    X_t, Y_t = get_windowed_tensors(N)
    res = run_training(X_t, Y_t, hidden_dims=deep_dims, epochs=100, lr=0.01)
    res["N"] = N
    window_results.append(res)


# =========================================================================
# 2. NETWORK DEPTH STUDY
# =========================================================================
print("Running Ablation 2: Network Depths (all N=10, 100 epochs, lr=0.01)...")
X_n10, Y_n10 = get_windowed_tensors(10)

depth_configs = [
    ("Shallow", [64], "input -> 64 -> output"),
    ("Medium", [128, 128], "input -> 128 -> 128 -> output"),
    ("Deep (current)", [128, 256, 128], "input -> 128 -> 256 -> 128 -> output"),
    ("Very Deep", [256, 512, 256, 128], "input -> 256 -> 512 -> 256 -> 128 -> output")
]

depth_results = []
for name, dims, arch_str in depth_configs:
    res = run_training(X_n10, Y_n10, hidden_dims=dims, epochs=100, lr=0.01)
    res["name"] = name
    res["arch"] = arch_str
    depth_results.append(res)


# =========================================================================
# 3. EPOCH COUNT STUDY
# =========================================================================
print("Running Ablation 3: Epoch Counts (50, 100, 200, 500 epochs, N=10, Deep network)...")
epoch_checkpoints = [50, 100, 200, 500]
epoch_res = run_training(X_n10, Y_n10, hidden_dims=deep_dims, epochs=500, lr=0.01, checkpoints=epoch_checkpoints)


# =========================================================================
# 4. LEARNING RATE STUDY
# =========================================================================
print("Running Ablation 4: Learning Rates (0.001, 0.005, 0.01, 0.05, 0.1)...")
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
lr_results = []

for lr_val in learning_rates:
    res = run_training(X_n10, Y_n10, hidden_dims=deep_dims, epochs=100, lr=lr_val)
    res["lr"] = lr_val
    lr_results.append(res)


# =========================================================================
# DISPLAY TABULAR RESULTS
# =========================================================================
print("\n" + "=" * 105)
print("                           MINECRAFT AI MODEL ABLATION STUDY RESULTS                           ")
print("=" * 105)

# Table 1: History Window Sizes
print("\n[TABLE 1] Ablation Study: History Window Size N (3-layer MLP: 128->256->128, 100 Epochs, lr=0.01)")
print("-" * 95)
print(f"| {'Window (N)':<10} | {'Input Size':<10} | {'Num Samples':<12} | {'Time (s)':<10} | {'Final Loss':<12} | {'Accuracy (%)':<14} |")
print("-" * 95)
for r in window_results:
    print(f"| N = {r['N']:<6} | {r['input_size']:<10} | {r['num_samples']:<12} | {r['training_time']:<10.3f} | {r['final_loss']:<12.4f} | {r['final_accuracy']:<13.2f}% |")
print("-" * 95)

# Table 2: Network Depths
print("\n[TABLE 2] Ablation Study: Network Depth (N=10, 100 Epochs, lr=0.01)")
print("-" * 108)
print(f"| {'Model Name':<15} | {'Architecture':<40} | {'Input':<6} | {'Time (s)':<9} | {'Final Loss':<11} | {'Accuracy (%)':<13} |")
print("-" * 108)
for r in depth_results:
    print(f"| {r['name']:<15} | {r['arch']:<40} | {r['input_size']:<6} | {r['training_time']:<9.3f} | {r['final_loss']:<11.4f} | {r['final_accuracy']:<12.2f}% |")
print("-" * 108)

# Table 3: Epoch Counts
print("\n[TABLE 3] Ablation Study: Epoch Counts (N=10, Deep: 128->256->128, lr=0.01)")
print("-" * 80)
print(f"| {'Epochs':<10} | {'Time Elapsed (s)':<18} | {'Loss at Checkpoint':<20} | {'Accuracy (%)':<16} |")
print("-" * 80)
for ep in epoch_checkpoints:
    cp = epoch_res["checkpoint_results"][ep]
    print(f"| {ep:<10} | {cp['time']:<18.3f} | {cp['loss']:<20.4f} | {cp['accuracy']:<15.2f}% |")
print("-" * 80)

# Table 4: Learning Rates
print("\n[TABLE 4] Ablation Study: Learning Rates (N=10, Deep: 128->256->128, 100 Epochs)")
print("-" * 95)
print(f"| {'Learning Rate':<15} | {'Input Size':<10} | {'Num Samples':<12} | {'Time (s)':<10} | {'Final Loss':<12} | {'Accuracy (%)':<14} |")
print("-" * 95)
for r in lr_results:
    print(f"| lr = {r['lr']:<10} | {r['input_size']:<10} | {r['num_samples']:<12} | {r['training_time']:<10.3f} | {r['final_loss']:<12.4f} | {r['final_accuracy']:<13.2f}% |")
print("-" * 95)

print("\n=== ABLATION STUDY COMPLETE ===")
