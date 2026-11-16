"""16-hundred-step independent comparison of online vs static/hybrid content clusters."""
from pathlib import Path
import subprocess
import sys
import json

lab = Path(__file__).resolve().parent
script = lab / "grid_study_runtime.py"
source = script.read_text()
anchor = '        if a.cinit == "freq": asg = freq_classes(K)'
assert source.count(anchor) == 1
source = source.replace(
    anchor,
    '        if a.cinit in ("context", "contextlate"): asg = np.load(os.path.join(DATA, "context_groups_6000.npy"))\n'
    '        elif a.cinit == "freq": asg = freq_classes(K)'
)
anchor = 'a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0'
assert source.count(anchor) == 1
source = source.replace(
    anchor,
    'a.clus and a.cinit in ("kmeans", "contextlate") and a.recl and step % a.recl == 0 '
    'and step > 0 and (a.cinit != "contextlate" or step >= 200)'
)
script.write_text(source)
result = lab / "long_training_compare.jsonl"
result.unlink(missing_ok=True)
for seed in (0, 1):
    for arm, flags in (
        ("baseline", []),
        ("dynamic", ["--clus", "64", "--cinit", "kmeans"]),
        ("context", ["--clus", "64", "--cinit", "context"]),
        ("contextlate", ["--clus", "64", "--cinit", "contextlate"]),
    ):
        cmd = [
            sys.executable, str(script), "--dev", "cpu",
            "--V", "6000", "--width", "32", "--steps", "1600",
            "--batch", "32", "--L", "64", "--threads", "2",
            "--heps", "1e-5", "--seed", str(seed),
            "--tag", f"long_{arm}_s{seed}", "--out", result.name, *flags,
        ]
        subprocess.run(cmd, check=True)
records = [json.loads(line) for line in result.read_text().splitlines()]
assert len(records) == 8
for seed in (0, 1):
    arms = {r["tag"].split("_")[1]: r for r in records if r["args"]["seed"] == seed}
    base = arms["baseline"]["scores"]["val"]
    for name, r in arms.items():
        print(f"LONGRUN seed={seed} arm={name} val={r['scores']['val']:.5f} "
              f"gain={base-r['scores']['val']:.5f} reassigns={len(r['regroup_records'])}", flush=True)
