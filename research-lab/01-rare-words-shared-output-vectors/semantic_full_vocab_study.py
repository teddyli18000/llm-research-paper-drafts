"""Pre-registered V50k semantic/static and late-online head-grouping comparators."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[41, 73, 109], required=True)
args = p.parse_args()

lab = Path(__file__).resolve().parent
runner = lab / "grid_study_runtime.py"
source = runner.read_text()
patches = [
    (
        '        if a.cinit == "freq": asg = freq_classes(K)',
        '        if a.cinit in ("context50", "contextlate50"): '
        'asg = np.load(os.path.join(DATA, "context_groups_50258.npy"))\n'
        '        elif a.cinit == "freq": asg = freq_classes(K)',
    ),
    (
        'a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0',
        'a.clus and a.cinit in ("kmeans", "contextlate50") and a.recl '
        'and step % a.recl == 0 and step > 0 '
        'and (a.cinit != "contextlate50" or step >= 200)',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

out = lab / f"semantic_full_vocab_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)

for name, init in (
    ("semantic_static", "context50"),
    ("semantic_late_online", "contextlate50"),
    ("frequency_static", "freq"),
):
    subprocess.run([
        sys.executable, str(runner), "--dev", "cpu", "--threads", "2",
        "--V", "50258", "--width", "64", "--layers", "2", "--heads", "4",
        "--L", "64", "--batch", "32", "--acc", "4", "--steps", "800",
        "--heps", "1e-5", "--hlr", "0.05", "--clr", "0.01",
        "--seed", str(args.seed),
        "--clus", "64", "--cinit", init, "--recl", "50",
        "--tag", f"semantic_{name}_s{args.seed}",
        "--out", out.name,
    ], check=True)

rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == 3 and len({x["tag"] for x in rows}) == 3
assert all(math.isfinite(x["scores"]["val"]) for x in rows)
vals = {row["tag"].removeprefix("semantic_").removesuffix(f"_s{args.seed}"): row["scores"]["val"] for row in rows}
for name, value in vals.items():
    print(f"SEMANTIC_V50K seed={args.seed} arm={name} val={value:.5f}", flush=True)
print(f"SEMANTIC_V50K seed={args.seed} late_minus_static={vals['semantic_late_online']-vals['semantic_static']:.5f}", flush=True)
