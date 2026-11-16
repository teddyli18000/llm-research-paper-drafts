"""Independent token-frequency bucket audit at V=16000, width=128."""
from pathlib import Path
import subprocess
import sys
import json
from statistics import mean

lab = Path(__file__).resolve().parent
script = lab / "grid_study_runtime.py"
source = script.read_text()
anchor = '        if a.cinit == "freq": asg = freq_classes(K)'
assert source.count(anchor) == 1
source = source.replace(
    anchor,
    '        if a.cinit == "context16": asg = np.load(os.path.join(DATA, "context_groups_16000.npy"))\n'
    '        elif a.cinit == "freq": asg = freq_classes(K)'
)
script.write_text(source)
result = lab / "frequency_bins_16k.jsonl"
result.unlink(missing_ok=True)

for seed in (0, 1):
    for arm, flags in (
        ("baseline", []),
        ("dynamic", ["--clus", "64", "--cinit", "kmeans"]),
        ("context", ["--clus", "64", "--cinit", "context16"]),
    ):
        cmd = [
            sys.executable, str(script),
            "--dev", "cpu", "--threads", "2",
            "--V", "16000", "--width", "128", "--layers", "2", "--heads", "4",
            "--steps", "400", "--batch", "32", "--hlr", "0.03",
            "--heps", "1e-5", "--clr", "0.005", "--seed", str(seed),
            "--tag", f"freq_{arm}_s{seed}", "--out", result.name, *flags,
        ]
        subprocess.run(cmd, check=True)
rows = [json.loads(line) for line in result.read_text().splitlines()]
assert len(rows) == 6
for seed in (0, 1):
    arms = {r["tag"].split("_")[1]: r for r in rows if r["args"]["seed"] == seed}
    print(f"SEED {seed}: baseline={arms['baseline']['scores']['val']} dynamic={arms['dynamic']['scores']['val']} context={arms['context']['scores']['val']}")
    for bucket, (baseline_loss, mass) in arms["baseline"]["fbins"].items():
        dynamic_loss = arms["dynamic"]["fbins"][bucket][0]
        context_loss = arms["context"]["fbins"][bucket][0]
        print(f"FREQ seed={seed} bucket={bucket} mass={mass:.4f} base={baseline_loss:.4f} dynamic={dynamic_loss:.4f} context={context_loss:.4f} contribution_dynamic={mass*(baseline_loss-dynamic_loss):.6f} contribution_context={mass*(baseline_loss-context_loss):.6f}",flush=True)
