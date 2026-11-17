"""Preregistered V16k/V50k head-optimizer transfer, three unseen seeds."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[41, 73, 109], required=True)
p.add_argument("--scale", type=int, choices=[16000, 50258], required=True)
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
        'p.add_argument("--ember_b2", type=float, default=0.999)',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    if a.ember_head:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '        opts.append(Ember([m.head], lr=a.hlr, beta2=a.ember_b2, eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
]
if args.scale == 16000:
    patches.append((
        '        if a.cinit == "freq": asg = freq_classes(K)',
        '        if a.cinit == "context16": asg = np.load(os.path.join(DATA, "context_groups_16000.npy"))\n'
        '        elif a.cinit == "freq": asg = freq_classes(K)',
    ))
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

if args.scale == 16000:
    width, steps, hlr, clr, accum = 128, 400, 0.03, 0.005, 1
else:
    width, steps, hlr, clr, accum = 64, 800, 0.05, 0.01, 4

out = lab / f"ember_transfer_V{args.scale}_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)

arm_specs = [
    ("adam_plain", False, False, 0.0, hlr, []),
    ("adam_dynamic", False, True, 0.0, hlr, []),
    ("ember_plain_b999", True, False, 0.999, 0.05, []),
    ("ember_plain_b095", True, False, 0.95, 0.05, []),
    ("ember_dynamic_b999", True, True, 0.999, 0.05, []),
    ("ember_dynamic_b095", True, True, 0.95, 0.05, []),
]
if args.scale == 16000:
    arm_specs.append(("context_static", False, False, 0.0, hlr, ["--clus", "64", "--cinit", "context16"]))

for name, use_ember, use_dynamic, beta2, lr, extras in arm_specs:
    flags = list(extras)
    if use_ember:
        flags += ["--ember_head", "1", "--ember_b2", str(beta2)]
    if use_dynamic:
        flags += ["--clus", "64", "--cinit", "kmeans", "--recl", "50"]
    subprocess.run([
        sys.executable, str(runner),
        "--dev", "cpu", "--threads", "2",
        "--V", str(args.scale), "--width", str(width), "--layers", "2", "--heads", "4",
        "--L", "64", "--batch", "32", "--acc", str(accum), "--steps", str(steps),
        "--heps", "1e-5", "--hlr", str(lr), "--clr", str(clr),
        "--seed", str(args.seed),
        "--tag", f"transfer_{name}_V{args.scale}_s{args.seed}",
        "--out", out.name, *flags,
    ], check=True)

rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == len(arm_specs)
assert len({x["tag"] for x in rows}) == len(rows)
assert all(math.isfinite(x["scores"]["val"]) for x in rows)
means = {name: r["scores"]["val"] for (name, *_), r in zip(arm_specs, rows)}
for name in means:
    print(f"TRANSFER V={args.scale} seed={args.seed} arm={name} CE={means[name]:.5f}", flush=True)
for b2 in ("b095", "b999"):
    print(f"EMBER_TRANSFER_GAIN V={args.scale} seed={args.seed} beta2={b2} "
          f"CE={means['ember_plain_'+b2]-means['ember_dynamic_'+b2]:.5f}", flush=True)
