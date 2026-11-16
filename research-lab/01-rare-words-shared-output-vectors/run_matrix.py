"""Execute matched baseline, clustering and Adam-state interventions."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["pilot", "confirm"], default="pilot")
a = p.parse_args()
if a.mode == "pilot":
    steps, seeds = 200, [0]
else:
    steps, seeds = 400, [0, 1, 2]
studies = [
    ("epsilon_baseline", [], "keep"),
    ("dynamic_keep", ["--clus", "64"], "keep"),
    ("dynamic_reset_c", ["--clus", "64"], "reset_c"),
    ("dynamic_reset_both", ["--clus", "64"], "reset_both"),
    ("one_recluster", ["--clus", "64", "--cstop", "0.26" if steps == 200 else "0.126"], "keep"),
    ("fixed_random", ["--clus", "64", "--cinit", "rand"], "keep"),
    ("fixed_frequency", ["--clus", "64", "--cinit", "freq"], "keep"),
]
output = ROOT / f"results_{a.mode}.jsonl"
output.unlink(missing_ok=True)
for seed in seeds:
    for name, flags, moment in studies:
        tag = f"{a.mode}_{name}_seed{seed}"
        cmd = [sys.executable, str(ROOT / "grid_study_runtime.py"), "--dev", "cpu",
               "--V", "6000", "--steps", str(steps), "--batch", "32", "--heps", "1e-5",
               "--seed", str(seed), "--moment", moment, "--tag", tag, "--out", output.name, "--threads", "2", *flags]
        subprocess.run(cmd, check=True)
records = [json.loads(line) for line in output.read_text().splitlines()]
print("\nSTUDY RESULT MATRIX (single validation loss; lower is better)")
for r in records:
    print(r["tag"], r["scores"]["val"], "reclusters", len(r["regroup_records"]))
