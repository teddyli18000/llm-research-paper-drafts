"""Matched head-optimizer challenge: tuned Coupled Adam, with/without dynamic groups."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
args = parser.parse_args()

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()

old = 'p.add_argument("--coupled", type=int, default=0)'
assert source.count(old) == 1
source = source.replace(
    old,
    old + '; p.add_argument("--cheps", type=float, default=1e-8)',
)
old = '((vv / (1 - a.b2 ** s.t)).sqrt() + a.eps)'
assert source.count(old) == 1
source = source.replace(old, '((vv / (1 - a.b2 ** s.t)).sqrt() + a.cheps)')
runner.write_text(source)

out = lab / f"coupled_challenge_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)

def train(tag, mode, lr, eps):
    flags = []
    if mode == "coupled":
        flags += ["--coupled", "1", "--cheps", str(eps)]
    if "dynamic" in tag:
        flags += ["--clus", "64", "--cinit", "kmeans", "--recl", "50"]
    subprocess.run([
        sys.executable, str(runner),
        "--dev", "cpu", "--threads", "1", "--V", "6000",
        "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", "32", "--steps", "400",
        "--heps", "1e-5", "--hlr", str(lr), "--clr", "0.02",
        "--seed", str(args.seed), "--tag", tag,
        "--out", out.name, *flags,
    ], check=True)

train(f"adam_plain_s{args.seed}", "adam", 0.1, 1e-5)
train(f"adam_dynamic_s{args.seed}", "adam", 0.1, 1e-5)
for eps in (1e-6, 1e-5):
    for lr in (0.02, 0.05, 0.1, 0.2):
        for grouping in ("plain", "dynamic"):
            tag = f"coupled_{grouping}_lr{lr:g}_eps{eps:g}_s{args.seed}"
            train(tag, "coupled", lr, eps)

rows = [json.loads(line) for line in out.read_text().splitlines()]
assert len(rows) == 18
by_tag = {row["tag"]: row["scores"]["val"] for row in rows}
baseline = by_tag[f"adam_plain_s{args.seed}"]
clustered = by_tag[f"adam_dynamic_s{args.seed}"]
print(f"ADAM seed={args.seed} plain={baseline:.5f} dynamic={clustered:.5f} gain={baseline-clustered:.5f}", flush=True)
for eps in (1e-6, 1e-5):
    for lr in (0.02, 0.05, 0.1, 0.2):
        pre = f"coupled_plain_lr{lr:g}_eps{eps:g}_s{args.seed}"
        dyn = f"coupled_dynamic_lr{lr:g}_eps{eps:g}_s{args.seed}"
        vp, vd = by_tag[pre], by_tag[dyn]
        print(f"COUPLED seed={args.seed} lr={lr:g} eps={eps:g} plain={vp:.5f} dynamic={vd:.5f} gain={vp-vd:.5f}", flush=True)
