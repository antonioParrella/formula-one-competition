"""The pipeline must survive a legacy Windows console.

Regression for KNOWN_ISSUES #3: `SurveyIndex.print_summary` prints a run of `─`
box-drawing characters, and on a default Windows console (cp1252) that raised
UnicodeEncodeError before the pipeline had done any work.
"""

import importlib.util
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

PIPELINE = Path("scripts/pipeline.py")

# Every non-ASCII string the pipeline's own output puts on stdout.
PIPELINE_OUTPUT = ["─" * 60, "─" * 85, "P1 → P10", "duplicate submission(s) — kept latest"]


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location("pipeline_under_test", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cp1252_console():
    """A stdout that behaves like a default Windows console."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", line_buffering=True)


def test_cp1252_console_cannot_take_the_output_unaided():
    # The failure this guards against — if this ever stops raising, the
    # pipeline fix below is no longer load-bearing.
    console = _cp1252_console()
    with pytest.raises(UnicodeEncodeError):
        console.write(PIPELINE_OUTPUT[0])
        console.flush()


def test_use_utf8_console_makes_the_output_printable(pipeline, monkeypatch):
    console = _cp1252_console()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", _cp1252_console())

    pipeline._use_utf8_console()
    for text in PIPELINE_OUTPUT:
        print(text)
    sys.stdout.flush()

    written = console.buffer.getvalue().decode("utf-8")
    for text in PIPELINE_OUTPUT:
        assert text in written


def test_survives_a_stream_that_cannot_be_reconfigured(pipeline, monkeypatch):
    class Unreconfigurable(io.StringIO):
        def reconfigure(self, **kwargs):
            raise OSError("not a real terminal")

    monkeypatch.setattr(sys, "stdout", Unreconfigurable())
    monkeypatch.setattr(sys, "stderr", Unreconfigurable())
    pipeline._use_utf8_console()  # must not raise


def test_run_pipeline_reconfigures_before_it_prints(pipeline):
    """The call has to be the first thing run_pipeline does — the old crash
    happened on the first table, well before any fetching."""
    source = PIPELINE.read_text(encoding="utf-8")
    body = source.split("def run_pipeline(", 1)[1]
    assert body.index("_use_utf8_console()") < body.index("print(")
