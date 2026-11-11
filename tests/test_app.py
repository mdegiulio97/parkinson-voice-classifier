"""The Streamlit dashboard renders the committed results without errors."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_renders_subject_level_results():
    at = AppTest.from_file(str(APP)).run(timeout=60)
    assert not at.exception
    assert any("subject" in s.value.lower() for s in at.subheader)
