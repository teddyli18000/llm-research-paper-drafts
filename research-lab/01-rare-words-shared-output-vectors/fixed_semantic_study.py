"""Compare fixed pretrained GPT-2 token groups on the same small model."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / "grid_study_runtime.py"
src = path.read_text()
src = src.replace(
    '        if a.cinit == "freq": asg = freq_classes(K)',
    '        if a.cinit == "pretrained": asg = np.load(os.path.join(DATA, "pretrained_groups_6000.npy"))\n        elif a.cinit == "freq": asg = freq_classes(K)',
)
path.write_text(src)
result = root / "semantic_fixed_study.jsonl"
result.unlink(missing_ok=True)
for seed in (0, 1, 2):
    cmd = [
        sys.executable, str(path), "--dev", "cpu",
        "--V", "6000", "--steps", "400", "--batch", "32",
        "--heps", "1e-5", "--seed", str(seed), "--clus", "64", "--cinit", "pretrained",
        "--tag", f"semantic_fixed_s{seed}", "--out", result.name, "--threads", "2",
    ]
    subprocess.run(cmd, check=True)
records = [json.loads(line) for line in result.read_text().splitlines()]
assert len(records) == 3
for record in records:
    print(record["tag"], "val",record["scores"]["val"])
