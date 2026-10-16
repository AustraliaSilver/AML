"""Read the A-matrices a `train_continual_aml.py` run prints, from its log.

    python experiments/parse_aml_matrix.py <log> [<log2> ...]

The trainer prints one line per evaluated cell:

    A[1,1] (mnist, routed): auto: 98.65% | oracle: 98.65% | live: 98.61% | route_acc 100.00%

A[i,j] is read at stage i (1-based, i.e. after training task i) and holds task
j's accuracy under three protocols. From that matrix this script derives the
usual continual-learning quantities:

    A_avg   mean over the seen tasks of the accuracy after the LAST task
    forget  mean over j < T of (task j right after j) - (task j after all T)
    BWT     -forget

Two protocol details matter when reading the output. The matrix is
*asymmetric on purpose*: `live` is measured with the task identity known and no
per-task restore, whereas `auto` first asks the router. And `evaluation` rows
printed earlier are not overwritten by later ones, so an empty cell means the
cell was never evaluated, not that it scored zero.

With two or more logs the script also prints a side-by-side comparison, which
is how a matched pair (e.g. the default snapshot against
`--snapshot-mode structural --freeze-trunk 1`) is read.

The encoding is sniffed from the BOM: Windows PowerShell's `*>` redirection
writes UTF-16LE while `Tee-Object` and `Out-File -Encoding utf8` write UTF-8.
"""

import re
import sys

CELL = re.compile(
    r"A\[(\d+),(\d+)\]\s*\(([^,]+),.*?\):\s*"
    r"auto:\s*([\d.]+)%\s*\|\s*oracle:\s*([\d.]+)%\s*\|\s*live:\s*([\d.]+)%"
    r"(?:\s*\|\s*route_acc\s*([\d.]+)%)?"
)

PROTOCOLS = ("oracle", "live", "auto")


def read_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse(path):
    """Return (cells, duplicates). The last print of a cell wins."""
    cells, seen = {}, {}
    for line in read_text(path).splitlines():
        m = CELL.search(line)
        if not m:
            continue
        i, j = int(m.group(1)), int(m.group(2))
        seen[(i, j)] = seen.get((i, j), 0) + 1
        cells[(i, j)] = {
            "task": m.group(3).strip(),
            "auto": float(m.group(4)),
            "oracle": float(m.group(5)),
            "live": float(m.group(6)),
            "route": float(m.group(7)) if m.group(7) else None,
        }
    return cells, seen


def metrics(cells, proto, T):
    final = [cells[(T, j)][proto] for j in range(1, T + 1) if (T, j) in cells]
    learnt = [cells[(j, j)][proto] for j in range(1, T) if (j, j) in cells]
    kept = [cells[(T, j)][proto] for j in range(1, T) if (T, j) in cells]
    forget = [a - b for a, b in zip(learnt, kept)]
    return {
        "acc": final,
        "A_avg": sum(final) / len(final) if final else float("nan"),
        "forget": sum(forget) / len(forget) if forget else float("nan"),
    }


def summarise(path, quiet=False):
    cells, dup = parse(path)
    if not cells:
        print(f"\n=== {path}\n    no A[i,j] cells found")
        return None
    T = max(max(i, j) for i, j in cells)
    order = [cells[(j, j)]["task"] if (j, j) in cells else f"task{j}" for j in range(1, T + 1)]
    twice = sorted(k for k, v in dup.items() if v > 1)

    out = {"path": path, "T": T, "order": order}
    if not quiet:
        print(f"\n=== {path}")
        print(f"    stages: {T} | cells re-printed: {twice or 'none'}")
    for proto in PROTOCOLS:
        st = metrics(cells, proto, T)
        out[proto] = st
        if quiet:
            continue
        print(f"\n  {proto.upper()}   (rows = after training task i, cols = task j)")
        print("      " + "".join(f"{t[:9]:>10}" for t in order))
        for i in range(1, T + 1):
            row = "".join(
                f"{cells[(i, j)][proto]:>10.2f}" if (i, j) in cells else f"{'-':>10}"
                for j in range(1, T + 1)
            )
            print(f"  T{i:<3}{row}")
        print(f"  A_avg={st['A_avg']:.2f}  forget={st['forget']:.2f}  "
              f"final: " + " ".join(f"{t}={a:.2f}" for t, a in zip(order, st["acc"])))
    return out


def compare(runs):
    print("\n" + "=" * 78)
    print(" COMPARISON")
    print("=" * 78)
    print(f"{'protocol':<9} | {'metric':<7} | " + " | ".join(
        f"{(r['path'].split(chr(92))[-1] if r else '?'):<28}" for r in runs))
    for proto in PROTOCOLS:
        for key in ("A_avg", "forget"):
            cells = []
            for r in runs:
                cells.append(f"{r[proto][key]:>28.2f}" if r else f"{'-':>28}")
            print(f"{proto:<9} | {key:<7} | " + " | ".join(cells))
    if len(runs) == 2 and runs[0] and runs[1]:
        print("\n delta (second - first):")
        for proto in PROTOCOLS:
            d_avg = runs[1][proto]["A_avg"] - runs[0][proto]["A_avg"]
            d_for = runs[1][proto]["forget"] - runs[0][proto]["forget"]
            print(f"  {proto:<7} A_avg {d_avg:+.2f}   forget {d_for:+.2f}")


def main(argv):
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 1
    runs = [summarise(p) for p in paths]
    if len(runs) > 1:
        compare(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
