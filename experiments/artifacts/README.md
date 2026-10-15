# experiments/artifacts

Raw output from the Phase-3 no-replay pilot (`experiments/noreplay_cl.py`).

| what | files |
| :-- | :-- |
| `logs_full_pilot*.txt` | stdout of the four pilot runs (1–4), the runs behind the `results_noreplay_full_pilot*.json` files that sit one level up |
| `results_*.json` | kept in `experiments/` proper, because `experiments/README.md` §3.1 cites them as the pilot's results |

Nothing here is required to run the code: `noreplay_cl.py` regenerates both
the logs and the JSON. They are tracked because a claim in the write-up points at
them, and a number whose source disappears cannot be checked.