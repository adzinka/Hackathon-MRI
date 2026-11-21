"""Real structural T1w (Morelli et al., sub-01, 3 T, defaced) through the unchanged ScanTwin T1w pipeline.

Run:  pytest -q tests/real_samples
The expected gate results are the ones recorded in external_data/real_samples/qc_report.json; if the pipeline changes,
regenerate the report (python scripts/run_real_samples.py) and review the difference.
"""
import numpy as np
import pytest

from twin_evidence import t1w as T
from twin_evidence.gates import classify

nib = pytest.importorskip("nibabel")
NAME = "sub-01_T1w.nii.gz"


@pytest.fixture(scope="module")
def run(real_file):
    s = T.load(real_file(NAME), name=NAME)
    m = T.measure(s)
    return s, m, T.gate(m)


def test_nifti_header_is_valid(real_file):
    img = nib.load(str(real_file(NAME)))
    h = img.header
    assert isinstance(img, nib.Nifti1Image)
    assert h.get_data_shape() == (208, 256, 256)
    assert h.get_zooms() == pytest.approx((1.0, 1.0, 1.0))
    assert int(h["qform_code"]) > 0 and int(h["sform_code"]) > 0
    assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
    a = np.asarray(img.dataobj)
    assert np.isfinite(a).all() and a.max() > 0
    mid = [a[a.shape[0] // 2], a[:, a.shape[1] // 2], a[:, :, a.shape[2] // 2]]     # the three views can be shown
    assert all(sl.std() > 0 for sl in mid)


def test_header_has_no_identifying_metadata(real_file, run):
    img = nib.load(str(real_file(NAME)))
    assert len(img.header.extensions) == 0
    s, _, _ = run
    assert s.removed == []


def test_pipeline_measures_what_it_can(run):
    s, m, g = run
    assert s.data.shape == (208, 256, 256) and s.voxel_mm3 == pytest.approx(1.0)
    assert not s.brain_mask                      # no brain mask supplied: automatic head mask
    assert m["motion_rating"] is not None and m["head_ml"] > 1000
    st = {x["id"]: x for x in g["gates"]}
    for gid in ("F27", "F28"):                   # needs a brain mask: N/A with a reason, never PASS
        assert st[gid]["status"] == "N/A" and "brain mask" in st[gid]["reason"]
    for gid in ("F32", "F35"):                   # needs a baseline scan / sidecars
        assert st[gid]["status"] == "N/A"


def test_result_matches_the_recorded_report(run, report):
    _, m, g = run
    rec = report["t1w"]
    assert {x["id"]: x["status"] for x in g["gates"]} == {x["id"]: x["status"] for x in rec["gates"]}
    assert g["status"] == rec["catalogue_status"]
    assert classify(g)["state"] == rec["state"]
    assert m["motion_rating"] == rec["metrics"]["motion_rating"]
