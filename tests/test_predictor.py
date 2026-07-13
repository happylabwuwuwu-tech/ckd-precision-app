"""Regression tests for ckd_pipeline.pkl.

These lock the model contract the Streamlit app depends on. They intentionally
encode *why* each property matters, so they fail loudly if the pkl is swapped
for one with a different feature set, output shape, or imputation behaviour.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PKL_PATH = Path(__file__).resolve().parents[1] / "ckd_pipeline.pkl"

# The exact set of feature keys app.py builds from the input form
# (SEC-01..05). If the form and the model ever disagree, predictions crash or
# silently reorder — this list is the guard.
APP_FEATURES = [
    # demographics + core labs entered outside SEC-02
    "crage", "male", "baseline_egfr", "nHBA1C", "nGLU", "nUPCR", "cciscore",
    # eGFR trajectory one-hot (Group 1 is the reference)
    "traj2", "traj3", "traj4",
    # SEC-02 lab_fields
    "nALT", "nAST", "nBUN", "nCHOL", "nTG", "nHDLC", "nLDLC", "nGA", "nUACR",
    # SEC-03 comorbidities
    "eDM", "ehtn", "echf", "ecva", "eami", "ecancer", "gerd",
    # SEC-04 medications
    "insulin", "acei_arb", "statin", "nsaid", "diuretics", "SGLT2",
    "beta_blocker", "ccb", "glp1", "Biguanid", "sulfonyl", "Meglitin",
    "AGIs", "TZD", "ddp4", "fibrate", "anticoagulant", "antacids", "antiplatelet",
]

THRESHOLD = 0.40  # must match app.py


@pytest.fixture(scope="module")
def pipeline():
    assert PKL_PATH.exists(), f"model file missing: {PKL_PATH}"
    return joblib.load(PKL_PATH)


@pytest.fixture
def model_features(pipeline):
    return list(pipeline.feature_names_in_)


def _row(model_features, **overrides):
    """A complete, in-range input row keyed by the model's own feature order."""
    base = {f: 0.0 for f in model_features}
    base.update({
        "crage": 60, "male": 1, "baseline_egfr": 60.0,
        "nHBA1C": 7.0, "nGLU": 100.0, "nUPCR": 0.5, "cciscore": 2,
    })
    base.update(overrides)
    return pd.DataFrame([base])[model_features]


# ── contract: structure ──────────────────────────────────────────────────────

def test_pipeline_has_expected_stages(pipeline):
    steps = dict(pipeline.named_steps)
    assert "imputer" in steps and "scaler" in steps and "model" in steps
    # median imputation is load-bearing for the "missing labs" UX decision
    assert steps["imputer"].strategy == "median"


def test_binary_classifier_with_45_features(pipeline):
    assert pipeline.n_features_in_ == 45
    assert list(pipeline.classes_) == [0, 1]


# ── contract: form ⇄ model feature parity (the invariant that guards SEC-01..05)

def test_app_form_features_exactly_match_model(model_features):
    assert set(APP_FEATURES) == set(model_features), (
        "app.py form fields and pipeline.feature_names_in_ diverged — "
        "predictions would crash or silently misalign"
    )
    assert len(APP_FEATURES) == 45


# ── contract: output shape / range ───────────────────────────────────────────

def test_predict_proba_in_unit_interval(pipeline, model_features):
    proba = pipeline.predict_proba(_row(model_features))[0]
    assert len(proba) == 2
    assert 0.0 <= proba[1] <= 1.0
    assert proba.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("egfr", [0.0, 5.0, 15.0, 60.0, 90.0, 150.0])
def test_extreme_egfr_does_not_crash_and_stays_in_range(pipeline, model_features, egfr):
    p = float(pipeline.predict_proba(_row(model_features, baseline_egfr=egfr))[0][1])
    assert 0.0 <= p <= 1.0


def test_all_zero_input_is_handled(pipeline, model_features):
    row = pd.DataFrame([{f: 0.0 for f in model_features}])[model_features]
    p = float(pipeline.predict_proba(row)[0][1])
    assert 0.0 <= p <= 1.0


# ── contract: missing values are median-imputed (documents & locks behaviour) ─

def test_missing_lab_is_imputed_not_crashed(pipeline, model_features):
    row = _row(model_features)
    row.loc[0, "nCHOL"] = np.nan
    p = float(pipeline.predict_proba(row)[0][1])
    assert 0.0 <= p <= 1.0  # imputer fills NaN, no crash


def test_training_medians_are_stable(pipeline, model_features):
    """Locks the pkl's learned medians so a silent model swap is caught."""
    stats = pipeline.named_steps["imputer"].statistics_
    idx = {f: i for i, f in enumerate(model_features)}
    assert stats[idx["nCHOL"]] == pytest.approx(155.0)
    assert stats[idx["nHBA1C"]] == pytest.approx(6.5351, abs=1e-3)
