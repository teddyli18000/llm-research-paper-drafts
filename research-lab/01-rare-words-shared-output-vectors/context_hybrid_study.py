"""Measure whether self-grouping adds value to strong same-data static context groups."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
script = root / "grid_study_runtime.py"
src = script.read_text()
src = src.replace(
    '        if a.cinit == "freq": asg = freq_classes(K)',
    '        if a.cinit == "context": asg = np.load(os.path.join(DATA, "context_groups_6000.npy"))\n        elif a.cinit == "freq": asg = freq_classes(K)',
)
src = src.replace(
    'p.add_argument("--seed", type=int, default=0)',
    'p.add_argument("--context_start", type=int, default=50); p.add_argument("--seed", type=int, default=0)',
)
src = src.replace(
    'a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0',
    'a.clus and a.cinit in ("kmeans", "context") and a.recl and step % a.recl == 0 and step >= a.context_start',
)
script.write_text(src)
result = root / "context_hybrid_study.jsonl"
result.unlink(missing_ok=True)
for seed in (0,1,2):
    for start in (50,200):
        subprocess.run([
            sys.executable,str(script),"--dev","cpu","--V","6000","--steps","400",
            "--batch","32","--heps","1e-5","--seed",str(seed),"--clus","64",
            "--cinit","context","--context_start",str(start),
            "--tag",f"context_hybrid_start{start}_s{seed}",
            "--out",result.name,"--threads","2",
        ],check=True)
rows = [json.loads(x) for x in result.read_text().splitlines()]
assert len(rows) == 6
for r in rows:
    print(r["tag"],"val",r["scores"]["val"],"reclusters",len(r["regroup_records"]))
