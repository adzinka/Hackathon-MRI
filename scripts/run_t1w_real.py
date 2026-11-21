"""Run the repo's existing T1w pipeline (unchanged) on downloaded real T1w NIfTI files. argv: repo nifti_dir out.json"""
import json
import sys
import time
from pathlib import Path

repo, nd, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo))
from twin_evidence import t1w  # noqa: E402

PAIRS = []
for s in ("sub-000103", "sub-000175", "sub-003617"):
    PAIRS += [(f"{s}_acq-standard", f"{s}_acq-{a}") for a in ("standard", "headmotion1", "headmotion2")]
PAIRS += [("sub-01_run-01", f"sub-01_run-0{r}") for r in (1, 2, 3, 4)]
PAIRS += [("sub-101_ses-1", f"sub-101_ses-{s}") for s in (1, 2)]

cache, res = {}, []


def get(stem):
    if stem not in cache:
        img = nd / f"{stem}_T1w.nii.gz"
        sc = nd / f"{stem}_T1w.json"
        side = json.loads(sc.read_text()) if sc.exists() else {}
        t = time.time()
        s = t1w.load(img, name=img.name, sidecar=side)
        m = t1w.measure(s)
        cache[stem] = (s, m, round(time.time() - t, 1))
    return cache[stem]


for b, f in PAIRS:
    s0, m0, _ = get(b)
    s1, m1, sec = get(f)
    m = m1 if b == f else t1w.measure(s1, s0, m0)
    g = t1w.gate(m)
    row = {"baseline": b, "scan": f, "seconds": sec, "shape": list(s1.data.shape), "zooms": list(s1.zooms),
           "removed_ident": s1.removed, "bg_masked": s1.bg_masked,
           **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
              if k in ("snr_wm", "cnr", "wm_cv_pct", "ghost_pct", "ghost_method", "motion_rating", "brain_ml", "gm_ml",
                       "wm_ml", "csf_ml", "cnr_pct_baseline", "reg_shift_mm", "image_change_sd", "protocol_diff",
                       "head_size_change_pct")},
           "status": g["status"], "gates": {x["id"]: x["status"] for x in g["gates"]}}
    if b != f:
        g0 = t1w.gate(m0)
        row["compare"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t1w.compare(m0, g0, m, g).items()
                          if k in ("kind", "title", "z", "pct", "status")}
    res.append(row)
    print(json.dumps(row), flush=True)
out.write_text(json.dumps(res, indent=1))
