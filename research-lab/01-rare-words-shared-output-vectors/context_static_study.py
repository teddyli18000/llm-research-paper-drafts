"""Run a static, same-data co-occurrence grouping against online K-means."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
script = root / "grid_study_runtime.py"
text = script.read_text()
text = text.replace(
    '        if a.cinit == "freq": asg = freq_classes(K)',
    '        if a.cinit == "context": asg = np.load(os.path.join(DATA, "context_groups_6000.npy"))\n        elif a.cinit == "freq": asg = freq_classes(K)',
)
script.write_text(text)
result = root / "context_static_study.jsonl"
result.unlink(missing_ok=True)
for seed in (0,1,2):
    subprocess.run([
        sys.executable,str(script),"--dev","cpu","--V","6000","--steps","400",
        "--batch","32","--heps","1e-5","--seed",str(seed),
        "--clus","64","--cinit","context",
        "--tag",f"context_static_s{seed}","--out",result.name,"--threads","2",
    ], check=True)
records = [json.loads(x) for x in result.read_text().splitlines()]
assert len(records) == 3
for r in records:
    print(r["tag"],"val",r["scores"]["val"])
