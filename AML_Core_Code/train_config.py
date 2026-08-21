"""
Cau hinh train cho tung nen tang khac nhau.
Su dung: python train_config.py --platform t4x2
"""

PLATFORMS = {
    "rtx3050": {
        "name": "RTX 3050 6GB Laptop",
        "embed_dim": 256,
        "num_layers": 2,
        "num_prototypes": 4,
        "hidden_dim": 512,
        "head_dim": 64,
        "num_heads": 4,
        "batch_size": 16,
        "seq_len": 64,
        "batches_per_epoch": 100,
        "epochs": 100,
        "lr": 8e-4,
        "max_chars": 300000,
        "use_teacher": False,
        "use_compile": False,
        "use_amp": True,
        "kl_weight": 0.0,
        "mse_weight": 0.0,
        "ce_weight": 1.0,
    },
    "t4x2": {
        "name": "T4x2 (2x 16GB = 32GB)",
        "embed_dim": 640,
        "num_layers": 4,
        "num_prototypes": 8,
        "hidden_dim": 2048,
        "head_dim": 256,
        "num_heads": 4,
        "batch_size": 32,
        "seq_len": 128,
        "batches_per_epoch": 200,
        "epochs": 200,
        "lr": 3e-4,
        "max_chars": 0,
        "use_teacher": True,
        "use_compile": True,
        "use_amp": True,
        "kl_weight": 0.0,
        "mse_weight": 0.0,
        "ce_weight": 1.0,
    },
    "tpu_v5e": {
        "name": "TPU v5e-8 (8 chips, 128GB total)",
        "embed_dim": 640,
        "num_layers": 4,
        "num_prototypes": 8,
        "hidden_dim": 2048,
        "head_dim": 256,
        "num_heads": 4,
        "batch_size": 128,
        "seq_len": 256,
        "batches_per_epoch": 500,
        "epochs": 300,
        "lr": 5e-4,
        "max_chars": 0,
        "use_teacher": True,
        "use_compile": True,
        "use_amp": True,
        "kl_weight": 0.0,
        "mse_weight": 0.0,
        "ce_weight": 1.0,
    },
}

def get_config(platform="rtx3050"):
    return PLATFORMS.get(platform, PLATFORMS["rtx3050"])

if __name__ == "__main__":
    import sys
    platform = sys.argv[1] if len(sys.argv) > 1 else "rtx3050"
    cfg = get_config(platform)
    print(f"\n=== {cfg['name']} ===")
    for k, v in cfg.items():
        if k != "name":
            print(f"  {k}: {v}")
