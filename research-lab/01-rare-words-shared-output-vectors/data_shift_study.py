"""Test whether gains reproduce on different FineWeb train slices (same held-out val)."""
from pathlib import Path
import subprocess
import sys
import json
from statistics import mean

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
code = runner.read_text()
anchor = 'p.add_argument("--seed", type=int, default=0)'
assert code.count(anchor) == 1
code = code.replace(anchor, 'p.add_argument("--data_offset", type=int, default=0); ' + anchor)
anchor = 'tr[:need]'
assert code.count(anchor) == 1
code = code.replace(anchor, 'tr[a.data_offset:a.data_offset + need]')
runner.write_text(code)

result = lab / "data_shift.jsonl"
result.unlink(missing_ok=True)

for offset in (0, 10_000_000, 40_000_000, 70_000_000):
    for seed in (0, 1):
        for arm, flags in (("baseline", []), ("dynamic", ["--clus", "64"])):
            cmd = [
                sys.executable, str(runner), "--dev", "cpu", "--threads", "2",
                "--V", "6000", "--steps", "400", "--batch", "32",
                "--heps", "1e-5", "--seed", str(seed), "--data_offset", str(offset),
                "--out", result.name, "--tag", f"offset{offset}_seed{seed}_{arm}", *flags,
            ]
            subprocess.run(cmd, check=True)

rows = [json.loads(x) for x in result.read_text().splitlines()]
assert len(rows) == 16
for offset in (0, 10_000_000, 40_000_000, 70_000_000):
    gains = []
    for seed in (0, 1):
        pair = {r["tag"].rsplit("_", 1)[1]: r["scores"]["val"] for r in rows if r["args"]["seed"] == seed and r["args"]["data_offset"] == offset}
        gains.append(pair["baseline"] - pair["dynamic"])
    print(f"DATA_OFFSET {offset}: seed_gains={gains}, mean_gain={mean(gains):.5f}", flush=True)
