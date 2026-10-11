"""Headless smoke test of the Streamlit app (needs the saved model)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "xgb_pre_transaction.joblib"

pytest.importorskip("streamlit")
pytest.importorskip("shap")
pytest.importorskip("xgboost")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="saved model not found")


def run_app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


def test_app_starts_without_errors():
    at = run_app()
    assert not at.exception
    assert any("Fraud probability" in m.label for m in at.metric)


def test_presets_give_opposite_decisions():
    at = run_app()

    at.button(key="preset_takeover").click().run()
    assert not at.exception
    assert any("FLAG" in e.value for e in at.error)

    at.button(key="preset_legit").click().run()
    assert not at.exception
    assert any("ALLOW" in s.value for s in at.success)
