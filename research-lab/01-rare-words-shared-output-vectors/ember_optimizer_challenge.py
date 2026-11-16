"""Matched official Ember head-only optimizer challenge with online group-sharing arms."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
args = p.parse_args()

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()

patches = [
    (
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F',
        'import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\nfrom ember import Ember',
    ),
    (
        'p.add_argument("--coupled", type=int, default=0)',
        'p.add_argument("--coupled", type=int, default=0); '
        'p.add_argument("--ember_head", type=int, default=0); '
        'p.add_argument("--ember_b2", type=float, default=0.999); '
        'p.add_argument("--ember_eps", type=float, default=1e-8)',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    if a.ember_head:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '        opts.append(Ember([m.head], lr=a.hlr, beta2=a.ember_b2, eps=a.ember_eps))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

out = lab / f"ember_challenge_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)

def train(name, ember, grouped, lr, beta2):
    flags = []
    if ember:
        flags += ["--ember_head", "1", "--ember_b2", str(beta2), "--ember_eps", "1e-8"]
    if grouped:
        flags += ["--clus", "64", "--cinit", "kmeans", "--recl", "50"]
    subprocess.run([
        sys.executable, str(runner), "--dev", "cpu", "--threads", "1",
        "--V", "6000", "--width", "32", "--layers", "2", "--heads", "2",
        "--L", "64", "--batch", "32", "--steps", "400",
        "--heps", "1e-5", "--hlr", str(lr), "--clr", "0.02",
        "--seed", str(args.seed), "--tag", name, "--out", out.name, *flags,
    ], check=True)

train(f"adam_plain_s{args.seed}", False, False, 0.1, 0.0)
train(f"adam_dynamic_s{args.seed}", False, True, 0.1, 0.0)
for beta2 in (0.95, 0.999):
    for lr in (0.01, 0.02, 0.05, 0.1):
        for grouping in ("plain", "dynamic"):
            name = f"ember_{grouping}_lr{lr:g}_b2{beta2:g}_s{args.seed}"
            train(name, True, grouping == "dynamic", lr, beta2)

records = [json.loads(x) for x in out.read_text().splitlines()]
assert len(records) == 18
by_tag = {r["tag"]: r["scores"]["val"] for r in records}
baseline = by_tag[f"adam_plain_s{args.seed}"]
dynamic = by_tag[f"adam_dynamic_s{args.seed}"]
print(f"ADAM seed={args.seed} plain={baseline:.5f} dynamic={dynamic:.5f} gain={baseline-dynamic:.5f}", flush=True)
for beta2 in (0.95, 0.999):
    for lr in (0.01, 0.02, 0.05, 0.1):
        pre = f"ember_plain_lr{lr:g}_b2{beta2:g}_s{args.seed}"
        dyn = f"ember_dynamic_lr{lr:g}_b2{beta2:g}_s{args.seed}"
        vp, vd = by_tag[pre], by_tag[dyn]
        print(f"EMBER seed={args.seed} lr={lr:g} beta2={beta2:g} plain={vp:.5f} dynamic={vd:.5f} gain={vp-vd:.5f}", flush=True)
