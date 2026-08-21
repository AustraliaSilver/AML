import os
import sys
import time
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))
from train import ConvAML_CIFAR_TrueDynamic, build_input_adapter


def get_testloader(dataset, root, batch_size=128):
    if dataset.startswith('cifar'):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        cls = torchvision.datasets.CIFAR100 if dataset == 'cifar100' else torchvision.datasets.CIFAR10
        testset = cls(root=root, train=False, download=True, transform=transform_test)
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        cls = torchvision.datasets.FashionMNIST if dataset == 'fashion_mnist' else torchvision.datasets.MNIST
        testset = cls(root=root, train=False, download=True, transform=transform_test)
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', default=['mnist', 'fashion_mnist'])
    parser.add_argument('--max-dim', type=int, default=48)
    parser.add_argument('--max-layers', type=int, default=8)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--task-id', type=int, default=-1, help="HtS task to activate")
    parser.add_argument('--hts-tasks', type=int, default=None)
    parser.add_argument('--hts-rank', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=8, help="Epochs used in training run (to reproduce final _D_ceiling)")
    parser.add_argument('--fixed-D', type=int, default=0)
    parser.add_argument('--fixed-L', type=int, default=0)
    parser.add_argument('--allow-delta0', action='store_true', help="Use trained delta[0] (backward transfer) for task 0 instead of zeroing it")
    parser.add_argument('--input-adapt', type=int, default=0, help="Reserved (kept for CLI compat); channel bridging handled internally by the model's gray_adapter")
    parser.add_argument('--base-channels', type=int, default=None, help="Intrinsic channel count of the shared base (default: read from checkpoint model_config)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _ckpt_full = torch.load(args.checkpoint, map_location=device)
    cfg = _ckpt_full.get('model_config', {}) if isinstance(_ckpt_full, dict) else {}
    if args.K is None:
        args.K = int(cfg.get('K', 32))
    if args.hts_tasks is None:
        args.hts_tasks = int(cfg.get('hts_tasks', 8) or 0)
    if args.hts_rank is None:
        args.hts_rank = int(cfg.get('hts_rank', 32))
    if args.base_channels is None:
        args.base_channels = int(cfg.get('base_channels', 3))
    if cfg.get('max_dim') is not None:
        args.max_dim = int(cfg['max_dim'])
    if cfg.get('max_layers') is not None:
        args.max_layers = int(cfg['max_layers'])
    if cfg.get('num_classes') is not None:
        num_classes = int(cfg['num_classes'])
    else:
        num_classes = 10
    print(f"Checkpoint config: {cfg}")

    model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=args.max_layers,
                                      in_channels=args.base_channels, num_classes=num_classes,
                                      hts_tasks=args.hts_tasks, hts_rank=args.hts_rank).to(device)
    model.fixed_D = args.fixed_D
    model.fixed_L = args.fixed_L
    if args.fixed_D == 0:
        phase_A_epochs = int(0.6 * args.epochs)
        progress_B = (args.epochs - 1 - phase_A_epochs) / max(1, args.epochs - phase_A_epochs)
        D_MIN = float(os.environ.get("AML_D_END", "24"))
        D_start = 0.95 * model.max_dim
        D_end = D_MIN
        model._D_ceiling = D_start - (D_start - D_end) * progress_B
    # Final-schedule D_ceiling (what the last epoch of training used). Capture
    # BEFORE any checkpoint override so prior tasks fall back to it.
    _schedule_D = float(getattr(model, '_D_ceiling', 0.0))
    # The saved state may be a mid-training best epoch (not the final epoch);
    # its architectural D_ceiling must be honored, otherwise such a state is
    # evaluated at the final epoch's lower D_ceiling and collapses. Fall back
    # to the schedule above for older checkpoints without the tag.
    _saved_D = float(_ckpt_full['_D_ceiling']) if (isinstance(_ckpt_full, dict) and '_D_ceiling' in _ckpt_full) else None
    _saved_D_task = int(_ckpt_full['_D_ceiling_task']) if (isinstance(_ckpt_full, dict) and '_D_ceiling_task' in _ckpt_full) else -1
    _D_map = _ckpt_full.get('_D_ceiling_map', {}) if isinstance(_ckpt_full, dict) else {}
    _ckpt = {k: v for k, v in _ckpt_full.items() if isinstance(v, torch.Tensor)}

    def _apply_D(task):
        # Use the per-task saved D_ceiling override (best epoch of each task).
        # Fall back to the legacy single-task tag, then to the final schedule.
        if _D_map and task in _D_map:
            model._D_ceiling = float(_D_map[task])
        else:
            model._D_ceiling = _saved_D if (_saved_D is not None and task == _saved_D_task) else _schedule_D
    missing, unexpected = model.load_state_dict(_ckpt, strict=False)
    if missing:
        print("MISSING KEYS:", missing)
    if unexpected:
        print("UNEXPECTED KEYS:", unexpected[:20])
    with torch.no_grad():
        for src_pre, dst_prefix in [("prototype_selector", "prototype_selectors"),
                                    ("structure_delta", "structure_deltas"),
                                    ("block_mode_head", "block_mode_heads")]:
            src_probe1 = f"{src_pre}.weight"
            src_probe2 = f"{src_pre}.0.weight"
            dst_probe = f"{dst_prefix}.0.weight"
            if (src_probe1 in _ckpt or src_probe2 in _ckpt) and dst_probe not in _ckpt:
                for t in range(model.num_task_blocks):
                    dst = getattr(model, dst_prefix)[t]
                    dst_params = {n: p for n, p in dst.named_parameters()}
                    dst_buffers = {n: b for n, b in dst.named_buffers()}
                    for (sk, sv) in _ckpt.items():
                        if sk.startswith(src_pre + "."):
                            sub = sk[len(src_pre)+1:]
                            if sub in dst_params:
                                dst_params[sub].copy_(sv)
                            elif sub in dst_buffers:
                                dst_buffers[sub].copy_(sv)
    if hasattr(model, 'hts') and model.hts is not None and args.allow_delta0:
        model.hts.allow_delta0 = True
    model.active_task = args.task_id
    model.eval()

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    for ds in args.datasets:
        loader = get_testloader(ds, data_root, args.batch_size)
        correct, total = 0, 0
        t0 = time.time()
        if args.task_id >= 0:
            model.active_task = args.task_id
            _apply_D(args.task_id)
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs, _, _, _ = model(inputs, tau=0.1)
                    correct += (outputs.argmax(-1) == targets).sum().item()
                    total += targets.size(0)
            print(f"{ds:14s} | task_id={args.task_id} | Acc: {correct/total*100:.2f}% | {time.time()-t0:.1f}s")
            continue

        # Auto task routing via the SHARED UNIVERSAL ROUTER on the RAW input:
        # per-sample argmax of router(x) decides which task's encoder/heads/delta
        # evaluate that sample. This never depends on any single encoder's
        # features, so it routes correctly at inference even though the router
        # was trained with each task's own features during its phase.
        model.active_task = -1
        trained = [i for i in range(args.hts_tasks) if float(model.hts.centroid_counts[i]) > 0]
        all_x, all_y, all_p = [], [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, _, _, diag = model(inputs, tau=0.1)
                rl = diag.get("router_logits")
                if rl is None:
                    print(f"{ds:14s} | router absent - could not auto-route | Acc: -")
                    break
                all_x.append(inputs)
                all_y.append(targets)
                all_p.append(rl.argmax(-1))
        if not all_x:
            continue
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_p = torch.cat(all_p, dim=0)
        correct, total = 0, 0
        with torch.no_grad():
            for t in trained:
                idx = (all_p == t).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                model.active_task = t
                _apply_D(t)
                chunk = 512
                for s in range(0, idx.numel(), chunk):
                    sl = idx[s:s + chunk]
                    outputs, _, _, _ = model(all_x[sl], tau=0.1)
                    correct += (outputs.argmax(-1) == all_y[sl]).sum().item()
                    total += sl.size(0)
        if total == 0:
            print(f"{ds:14s} | router absent - falling back to task 0 | Acc: -")
            continue
        print(f"{ds:14s} | auto per-sample routing | Acc: {correct/total*100:.2f}% | {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()