"""TRUSTY / WARNING / ERROR: the catalogue decision rules under their on-screen names, plus the minimum-set rule."""
from twin_evidence.gates import classify, evaluate_mrs, evaluate_t1w

GOOD = {"motion_rating": 1, "cnr_pct_baseline": 95.0, "wm_cv_pct": 5.0, "reg_shift_mm": 0.5, "protocol_diff": 0.0}


def test_all_pass_is_trusty():
    c = classify(evaluate_t1w(GOOD))
    assert c["state"] == "TRUSTY" and c["verdict"] == "update"
    assert {x["category"]: x["status"] for x in c["categories"]} == {
        "Motion": "TRUSTY", "Image quality": "TRUSTY", "Protocol": "TRUSTY", "Comparison between scans": "TRUSTY"}


def test_warn_is_warning_and_fail_is_error():
    assert classify(evaluate_t1w({**GOOD, "motion_rating": 2}))["state"] == "WARNING"
    c = classify(evaluate_t1w({**GOOD, "wm_cv_pct": 25.0}))
    assert c["state"] == "ERROR" and c["verdict"] == "reject" and any("F28" in r for r in c["reasons"])


def test_missing_input_is_never_trusty_and_never_error():
    c = classify(evaluate_t1w({**GOOD, "motion_rating": None}))
    assert c["state"] == "WARNING" and c["unchecked"] == ["F26"]
    assert {x["category"]: x["status"] for x in c["categories"]}["Motion"] == "N/A"


def test_a_dicom_error_blocks_trusty():
    dicom = {"findings": [{"id": "Series complete", "catalogue_id": None, "label": "Series complete", "status": "FAIL",
                           "value": "15 slices; 1 missing", "reason": "a slice is missing", "kind": "file"}]}
    c = classify(evaluate_t1w(GOOD), dicom)
    assert c["state"] == "ERROR"
    assert {x["category"]: x["status"] for x in c["categories"]}["DICOM integrity"] == "ERROR"


def test_mrs_verdicts_map_one_to_one():
    """The MRS gates are only wrapped, never changed: Gate 1 numbers stay as they are."""
    for m, state in [({"fwhm_hz": 5, "snr": 40}, "TRUSTY"), ({"fwhm_hz": 15, "snr": 40}, "WARNING"),
                     ({"fwhm_hz": 25, "snr": 40}, "ERROR")]:
        g = evaluate_mrs(m)
        assert classify(g, minimum=())["state"] == state
        assert classify(g, minimum=())["verdict"] == g["verdict"]
