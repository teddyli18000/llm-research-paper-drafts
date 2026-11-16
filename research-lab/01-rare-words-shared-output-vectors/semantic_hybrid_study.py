"""Test whether pretrained semantic groups benefit from subsequent online regrouping."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
p = root / "grid_study_runtime.py"
s = p.read_text()
s = s.replace(
    '        if a.cinit == "freq": asg = freq_classes(K)',
    '        if a.cinit == "pretrained": asg = np.load(os.path.join(DATA, "pretrained_groups_6000.npy"))\n        elif a.cinit == "freq": asg = freq_classes(K)',
)
s = s.replace(
    'p.add_argument("--seed", type=int, default=0)',
    'p.add_argument("--pretrained_start", type=int, default=50); p.add_argument("--seed", type=int, default=0)',
)
s = s.replace(
    'a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0',
    'a.clus and a.cinit in ("kmeans", "pretrained") and a.recl and step % a.recl == 0 and step >= a.pretrained_start',
)
p.write_text(s)
result = root / "semantic_hybrid_study.jsonl"
result.unlink(missing_ok=True)
for seed in (0, 1, 2):
    for start in (50, 200):
        cmd = [
            sys.executable, str(p), "--dev", "cpu", "--V", "6000",
            "--steps", "400", "--batch", "32", "--heps", "1e-5",
            "--clus", "64", "--cinit", "pretrained", "--pretrained_start", str(start),
            "--seed", str(seed), "--tag", f"pretrained_hybrid_start{start}_s{seed}",
            "--out", result.name, "--threads", "2",
        ]
        subprocess.run(cmd, check=True)
records = [json.loads(line) for line in result.read_text().splitlines()]
assert len(records) == 6
for record in records:
    print(record["tag"], "val", record["scores"]["val"], "reclusters", len(record["regroup_records"]))
