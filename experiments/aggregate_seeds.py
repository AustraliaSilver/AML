"""Aggregate multi-seed JSONs from experiments/exp_zero_per_task.py.

    python experiments/aggregate_seeds.py out_s0.json out_s1.json out_s2.json

Prints mean and spread per mechanism, because a single-seed table is not a
result: the OGP adjudication showed the naive baseline alone moving 34.37 ->
53.27 % across environments.  Mechanics: for each method, the mean, the min/max
range and the sample standard deviation across the seeds given. Byte columns
are taken from the first file that has the method (they do not vary by seed).
"""
import json
import statistics
import sys


def human(n):
    if n == 0:
        return "0"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1048576:.3f} MiB"


def main(paths):
    per_method = {}
    n_init = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            blob = json.load(fh)
        n_init.append(blob["acc_m_init"])
        for r in blob["results"]:
            per_method.setdefault(r["method"], []).append(r)

    print(f"seeds: {len(paths)}   MNIST-after-task-1 (init): "
          f"{statistics.mean(n_init):.2f}% (range {min(n_init):.2f}-{max(n_init):.2f})")
    hdr = (f"{'mechanism':<20} | {'bytes/task':>10} | {'fixed':>9} | "
           f"{'MNIST':>15} | {'TASK2':>15} | {'forget':>15}")
    print(hdr)
    print("-" * len(hdr))
    for m, rows in per_method.items():
        mem = statistics.mean(r["acc_m"] for r in rows)
        tef = statistics.mean(r["acc_f"] for r in rows)
        fog = statistics.mean(r["forget"] for r in rows)
        sd_m = statistics.stdev([r["acc_m"] for r in rows]) if len(rows) > 1 else 0.0
        sd_f = statistics.stdev([r["acc_f"] for r in rows]) if len(rows) > 1 else 0.0
        sd_g = statistics.stdev([r["forget"] for r in rows]) if len(rows) > 1 else 0.0
        r0 = rows[0]
        bpt = human(r0["per_task"])
        fx = human(r0["fixed"])
        print(f"{m:<20} | {bpt:>10} | {fx:>9} | {mem:>6.2f} ±{sd_m:>5.2f} | "
              f"{tef:>6.2f} ±{sd_f:>5.2f} | {fog:>6.2f} ±{sd_g:>5.2f}")
    # ranked by the ACC analogue: mean of the two task accuracies
    print("\n ranked by ACC = mean(MNIST, TASK2)")
    ranked = sorted(per_method.items(),
                    key=lambda kv: -(statistics.mean(r["acc_m"] for r in kv[1])
                                     + statistics.mean(r["acc_f"] for r in kv[1])) / 2)
    for m, rows in ranked:
        acc = (statistics.mean(r["acc_m"] for r in rows)
               + statistics.mean(r["acc_f"] for r in rows)) / 2
        print(f"  {acc:6.2f}  {m}")


if __name__ == "__main__":
    main(sys.argv[1:])
