"""Matched tied-input-output token-table study at 400 and 1600 steps."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
p.add_argument("--steps", type=int, choices=[400, 1600], required=True)
args = p.parse_args()

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()
old = '        if a.cinit == "freq": asg = freq_classes(K)'
assert source.count(old) == 1
source = source.replace(
    old,
    '        if a.cinit == "context": asg = np.load(os.path.join(DATA, "context_groups_6000.npy"))\n'
    '        elif a.cinit == "freq": asg = freq_classes(K)'
)
runner.write_text(source)

out = lab / f"tied_s{args.seed}_steps{args.steps}.jsonl"
out.unlink(missing_ok=True)

for arm, flags in (
    ("plain", []),
    ("dynamic", ["--clus", "64", "--cinit", "kmeans", "--recl", "50"]),
    ("context", ["--clus", "64", "--cinit", "context"]),
):
    subprocess.run([
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", "32", "--steps", str(args.steps),
        "--tie", "1", "--heps", "1e-5", "--hlr", "0.1", "--clr", "0.02",
        "--seed", str(args.seed), "--tag", f"tied_{arm}_s{args.seed}_t{args.steps}",
        "--out", out.name, *flags,
    ], check=True)

rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == 3
values = {r["tag"].split("_")[1]: r["scores"]["val"] for r in rows}
print(
    f"TIED seed={args.seed} steps={args.steps} "
    f"plain={values['plain']:.5f} dynamic={values['dynamic']:.5f} "
    f"context={values['context']:.5f} "
    f"online_gain={values['plain']-values['dynamic']:.5f} "
    f"static_gain={values['plain']-values['context']:.5f}",
    flush=True,
)
