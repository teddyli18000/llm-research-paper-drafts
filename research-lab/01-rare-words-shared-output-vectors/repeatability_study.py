"""Measure same-seed numerical reproducibility across CPU thread settings."""
import json
from pathlib import Path
import subprocess
import sys
from statistics import pstdev

root = Path(__file__).resolve().parent
script = root / "grid_study_runtime.py"
out = root / "repeatability.jsonl"
out.unlink(missing_ok=True)
for threads in (1, 2):
    for repeat in range(3):
        cmd = [
            sys.executable, str(script),
            "--dev", "cpu", "--threads", str(threads),
            "--V", "16000", "--width", "128", "--layers", "2", "--heads", "4",
            "--steps", "400", "--batch", "32", "--hlr", "0.03", "--clr", "0.005",
            "--heps", "1e-5", "--seed", "0",
            "--clus", "64", "--cinit", "kmeans",
            "--tag", f"repeat_t{threads}_r{repeat}", "--out", out.name,
        ]
        subprocess.run(cmd, check=True)
rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == 6
for threads in (1, 2):
    losses = [r["scores"]["val"] for r in rows if r["args"]["threads"] == threads]
    print(f"REPEAT threads={threads} losses={losses} range={max(losses)-min(losses):.6f} sd={pstdev(losses):.6f}", flush=True)
