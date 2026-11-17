"""Locked 5-arm new-seed V50k evaluation against independent FineWeb shard002."""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, choices=[809, 811, 821], required=True)
args = p.parse_args()

lab = Path(__file__).resolve().parent
data = lab / "data"
test_path = hf_hub_download(
    repo_id="kjj0/fineweb10B-gpt2", repo_type="dataset",
    filename="fineweb_train_000002.bin",
)
shutil.copyfile(test_path, data / "fineweb_train_000002.bin")

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
        'p.add_argument("--projector_alpha", type=float, default=0.0); '
        'p.add_argument("--ember_b2", type=float, default=0.999)',
    ),
    (
        '        K = max(a.clus, 1); s.Ch = nn.Parameter(torch.zeros(K, d))',
        '        K = 64 if a.projector_alpha else max(a.clus, 1); '
        's.Ch = nn.Parameter(torch.zeros(K, d))',
    ),
    (
        '        if a.cinit == "freq": asg = freq_classes(K)',
        '        if a.cinit == "context50": asg = np.load(os.path.join(DATA,"context_groups_50258.npy"))\n'
        '        elif a.cinit == "freq": asg = freq_classes(K)',
    ),
    (
        '    if a.coupled:\n        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
        '    if a.ember_head:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]\n'
        '        opts.append(Ember([m.head], lr=a.hlr, beta2=a.ember_b2, eps=1e-8))\n'
        '    if a.coupled:\n'
        '        groups = [g for g in groups if g["params"][0] is not m.head]; opts.append(CoupledAdam([m.head], a.hlr))',
    ),
    (
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
        '        if a.projector_alpha and step and step % 50 == 0:\n'
        '            with torch.no_grad():\n'
        '                m.ah.copy_(kmeans(m.head.detach(),64))\n'
        '        if a.cls and a.clsby == "kmeans" and a.recl and step % a.recl == 0 and step > 0: m.recluster_cls()',
    ),
    (
        '        for o in opts: o.step()',
        '        if a.projector_alpha:\n'
        '            with torch.no_grad():\n'
        '                g=m.head.grad\n'
        '                pooled=torch.zeros(64,g.shape[1],device=g.device).index_add_(0,m.ah,g)\n'
        '                counts=torch.bincount(m.ah,minlength=64).clamp_min(1)\n'
        '                g.add_(pooled[m.ah]/counts[m.ah,None],alpha=a.projector_alpha)\n'
        '        for o in opts: o.step()',
    ),
    (
        'val = look[np.stack([np.asarray(va[x:x + a.L]) for x in np.linspace(0, len(va) - a.L - 1, a.nval).astype(np.int64)]).astype(np.int64)]',
        'val = look[np.stack([np.asarray(va[x:x + a.L]) for x in np.linspace(0, len(va) - a.L - 1, a.nval).astype(np.int64)]).astype(np.int64)]\n'
        'ta=np.memmap(os.path.join(DATA,"fineweb_train_000002.bin"),dtype=np.uint16,mode="r",offset=1024)\n'
        'test=look[np.stack([np.asarray(ta[x:x+a.L]) '
        'for x in np.linspace(0,len(ta)-a.L-1,a.nval).astype(np.int64)]).astype(np.int64)]',
    ),
    (
        'VX = torch.tensor(val).to(D); VB = obin[val[:, 1:].reshape(-1)]; VF = fbin[val[:, 1:].reshape(-1)]',
        'VX = torch.tensor(val).to(D); VB = obin[val[:, 1:].reshape(-1)]; VF = fbin[val[:, 1:].reshape(-1)]\n'
        'TX=torch.tensor(test).to(D); TB=obin[test[:,1:].reshape(-1)]; TF=fbin[test[:,1:].reshape(-1)]',
    ),
    (
        'def main():',
        'def evaluate_test(m):\n'
        '    m.eval()\n'
        '    with torch.no_grad():\n'
        '        per=torch.cat([m.nll(TX[i:i+50,:-1],TX[i:i+50,1:]) '
        'for i in range(0,len(TX),50)]).cpu().numpy()\n'
        '    m.train()\n'
        '    return per\n\n'
        'def main():',
    ),
    (
        '    per = evaluate(m)\n    r = {"val": float(per.mean())}',
        '    per = evaluate(m)\n'
        '    per_test=evaluate_test(m)\n'
        '    r = {"val": float(per.mean()), "test":float(per_test.mean())}',
    ),
    (
        '    if m.uh:\n        m.crev = True;',
        '    tbins={}\n'
        '    for i,n in enumerate(BN+["unk"]):\n'
        '        sel=TB==i\n'
        '        if sel.any(): tbins[n]=[round(float(per_test[sel].mean()),4),round(float(sel.mean()),4)]\n'
        '    tfbins={}\n'
        '    for i,n in enumerate(FN+["unk"]):\n'
        '        sel=TF==i\n'
        '        if sel.any(): tfbins[n]=[round(float(per_test[sel].mean()),4),round(float(sel.mean()),4)]\n'
        '    if m.uh:\n        m.crev = True;',
    ),
    (
        'bins=bins, fbins=fb)',
        'bins=bins, fbins=fb, test_bins=tbins, test_fbins=tfbins)',
    ),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
runner.write_text(source)

out = lab / f"locked_external_seed{args.seed}.jsonl"
out.unlink(missing_ok=True)
specs = [
    ("adam_dynamic", False, True, False, False),
    ("ember_plain", True, False, False, False),
    ("ember_centroid", True, True, False, False),
    ("ember_projector", True, False, True, False),
    ("adam_static_context", False, False, False, True),
]

for name, use_ember, use_centroid, use_proj, use_context in specs:
    flags = []
    if use_ember:
        flags += ["--ember_head", "1", "--ember_b2", "0.999"]
    if use_centroid:
        flags += ["--clus", "64", "--cinit", "kmeans", "--recl", "50"]
    if use_proj:
        flags += ["--projector_alpha", "2.0"]
    if use_context:
        flags += ["--clus", "64", "--cinit", "context50"]
    subprocess.run([
        sys.executable, str(runner),
        "--dev", "cpu", "--threads", "2", "--V", "50258", "--width", "64",
        "--layers", "2", "--heads", "4", "--batch", "32", "--L", "64",
        "--steps", "800", "--acc", "4", "--heps", "1e-5",
        "--hlr", "0.05", "--clr", "0.01", "--seed", str(args.seed),
        "--tag", f"external_{name}_s{args.seed}",
        "--out", out.name, *flags,
    ], check=True)

rows = [json.loads(x) for x in out.read_text().splitlines()]
assert len(rows) == 5 and len({r["tag"] for r in rows}) == 5
assert all(math.isfinite(r["scores"]["val"]) and math.isfinite(r["scores"]["test"]) for r in rows)
for (name, *_), record in zip(specs, rows):
    print(f"EXTERNAL_HOLDOUT seed={args.seed} arm={name} "
          f"original_val={record['scores']['val']:.5f} "
          f"external_test={record['scores']['test']:.5f} "
          f"train_s={record['train_s']:.1f}", flush=True)
