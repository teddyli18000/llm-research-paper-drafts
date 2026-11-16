"""Build a modified copy of the original grid.py for intervention tests."""
from pathlib import Path

LAB = Path(__file__).resolve().parent
SOURCE = LAB.parents[1] / "01-rare-words-shared-output-vectors" / "grid.py"
source = SOURCE.read_text()
patches = [
    ('DATA = os.path.join(HERE, "..", "llm-per-word-weights", "data"); EOT = 50256',
     'DATA = os.path.join(HERE, "data"); EOT = 50256'),
    ('p.add_argument("--save", default="", help="save final head rows / assignments to this .pt path")',
     'p.add_argument("--seed", type=int, default=0); p.add_argument("--moment", choices=["keep", "reset_c", "reset_both"], default="keep")\np.add_argument("--save", default="", help="save final head rows / assignments to this .pt path")'),
    ('torch.manual_seed(0); D = torch.device(a.dev)', 'torch.manual_seed(a.seed); D = torch.device(a.dev)'),
    ('    t0 = time.time(); X = torch.tensor(rows).to(D)',
     '    regroup_records = []\n    t0 = time.time(); X = torch.tensor(rows).to(D)'),
    ('        if a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0 and step < a.cstop * a.steps: m.recluster()',
     '''        if a.clus and a.cinit == "kmeans" and a.recl and step % a.recl == 0 and step > 0 and step < a.cstop * a.steps:
            old_head = m.H().detach().clone()
            old_assignment = m.ah.clone()
            m.recluster()
            error = (m.H() - old_head).abs().max().item()
            assert error < 1e-5
            churn = (m.ah != old_assignment).float().mean().item()
            regroup_records.append([step, churn, error])
            if a.moment != "keep":
                params = [m.Ch] if a.moment == "reset_c" else [m.Ch, m.head]
                for p in params:
                    st = opts[-1].state[p]
                    st["exp_avg"].zero_()
                    st["exp_avg_sq"].zero_()'''),
    ('    res = dict(tag=a.tag, args=vars(a), tokens=need',
     '    res = dict(regroup_records=regroup_records, tag=a.tag, args=vars(a), tokens=need'),
]
for old, new in patches:
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)
(LAB / "grid_study_runtime.py").write_text(source)
print("Instrumented original experiment copy; upstream grid.py unchanged")
