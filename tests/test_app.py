"""Streamlit-level regression tests for the three fixes in commit 8f0e191:

1. CKD staging covers G1/G2 (eGFR >= 60 is not mislabeled G3a)
2. Health-education block never silently disappears (fallback for G1/G2/G5)
3. Lab inputs are required — prediction is blocked, not silently imputed

Run headless via streamlit.testing; no browser needed.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _fresh():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception  # empty ElementList() == no exception
    return at


def _fill_labs_and_egfr(at, egfr):
    for ni in at.number_input:
        if ni.value is None:
            ni.set_value(20.0)
    for ni in at.number_input:
        if ni.label and "eGFR" in ni.label:
            ni.set_value(egfr)
    return at.run()


def test_cold_load_has_no_exception():
    at = _fresh()  # catches bad kwargs (value=None / placeholder) and markup errors
    assert len(at.number_input) > 0


def test_running_with_empty_labs_is_blocked():
    at = _fresh()
    at.button[0].click().run()
    assert any("檢驗值" in e.value for e in at.error), "missing-lab guard did not fire"
    assert at.session_state["result"] is None, "must not predict with empty labs"


def test_prediction_runs_once_labs_are_filled():
    at = _fresh()
    _fill_labs_and_egfr(at, 60.0)
    at.button[0].click().run()
    assert not at.exception, at.exception  # empty ElementList() == no exception
    res = at.session_state["result"]
    assert res is not None
    assert 0.0 <= res["prob_decline"] <= 1.0


@pytest.mark.parametrize("egfr,stage", [(90.0, "G1"), (75.0, "G2"), (10.0, "G5")])
def test_education_block_renders_for_uncatalogued_stages(egfr, stage):
    """G1/G2/G5 have no explicit edu entry; the fallback must still render."""
    at = _fresh()
    _fill_labs_and_egfr(at, egfr)
    at.button[0].click().run()
    pv = [b for b in at.button if b.label == "Patient View"]
    assert pv, "Patient View button missing"
    pv[0].click().run()
    assert not at.exception, at.exception  # empty ElementList() == no exception
    md = " ".join(m.value for m in at.markdown)
    assert "衛教" in md, f"education block silently missing for {stage}"
    assert stage in md, f"stage label {stage} not shown"
