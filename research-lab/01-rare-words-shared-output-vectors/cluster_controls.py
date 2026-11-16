"""Isolate learned content-driven regrouping from arbitrary reparameterization."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
script = root / "grid_study_runtime.py"
src = script.read_text()
src = src.replace(
    'p.add_argument("--seed", type=int, default=0)',
    'p.add_argument("--group_mode", choices=["kmeans", "random_refresh", "same_membership", "warmstart"], default="kmeans"); p.add_argument("--seed", type=int, default=0)'
)
src = src.replace(
    '        asg = kmeans(R, K)\n        cm = torch.zeros(K, R.shape[1], device=R.device)',
    '''        if a.group_mode == "random_refresh":
            asg = torch.randperm(len(R), device=R.device) % K
        elif a.group_mode == "same_membership":
            asg = s.ah.clone()
        elif a.group_mode == "warmstart":
            old_means = torch.zeros(K, R.shape[1], device=R.device).index_add_(0, s.ah, R)
            counts = torch.bincount(s.ah, minlength=K).clamp(min=1)
            mu = old_means / counts[:, None]
            for _ in range(10):
                labels = torch.cdist(R, mu).argmin(1)
                counts = torch.bincount(labels, minlength=K)
                mu = torch.where(
                    counts[:, None] > 0,
                    torch.zeros_like(mu).index_add_(0, labels, R) / counts.clamp(min=1)[:, None],
                    mu,
                )
            asg = torch.cdist(R, mu).argmin(1)
        else:
            asg = kmeans(R, K)
        cm = torch.zeros(K, R.shape[1], device=R.device)'''
)
script.write_text(src)
results = root / "cluster_controls.jsonl"
results.unlink(missing_ok=True)
for seed in (0, 1, 2):
    for mode in ("random_refresh", "same_membership", "warmstart"):
        cmd = [
            sys.executable, str(script), "--dev", "cpu", "--V", "6000",
            "--steps", "400", "--batch", "32", "--heps", "1e-5",
            "--clus", "64", "--group_mode", mode, "--seed", str(seed),
            "--tag", f"cluster_{mode}_s{seed}", "--out", results.name, "--threads", "2",
        ]
        subprocess.run(cmd, check=True)
rows = [json.loads(line) for line in results.read_text().splitlines()]
assert len(rows) == 9
for r in rows:
    print(r["tag"], "val", r["scores"]["val"], "reclusters", len(r["regroup_records"]))
