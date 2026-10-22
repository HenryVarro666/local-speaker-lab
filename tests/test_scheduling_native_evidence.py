"""Do not pass a native-tool run on speech that still states an obsolete fact."""

import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Friday October 23rd at 4:30 PM Central, reference BK-0001.", True),
        ("Friday at four thirty PM, reference B K zero zero zero one.", True),
        ("Friday at four thirty P M, reference B K zero zero zero one.", True),
        ("Friday at four thirty PM, reference B K triple zero one.", True),
        ("Friday at four thirty PM, reference B K three zeros followed by one.", True),
        ("Friday at four thirty PM, reference B K 3 zeros followed by 1.", True),
        ("Friday at four thirty PM, reference B K two zeros followed by one.", False),
        ("Friday at four thirty PM, reference B K double zero one.", False),
        ("Hello, how can I help you today?", False),
        ("Thursday at 4:30 PM, reference BK-0001.", False),
        ("Friday at 3:30 PM, reference BK-0001.", False),
        ("Friday at 4:30 AM, reference BK-0001.", False),
        ("Friday at 4:30 PM, reference BK-0002.", False),
        ("Friday at 4:30 PM, reference BK-00012.", False),
        ("Friday at 4:30 PM, reference ABK-0001.", False),
        ("Friday at 4:30 PM, reference BK0001 then BK0002.", False),
    ],
)
def test_native_fact_guard(text, expected):
    path = Path(__file__).resolve().parents[1] / "scripts/scheduling_smoke.py"
    spec = importlib.util.spec_from_file_location("native_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.native_fact_matches(text, "2026-10-23T16:30:00-05:00", "BK-0001")
        is expected
    )


@pytest.mark.parametrize('text,expected', [
    ('Friday October 23rd at 4:30 PM, BK0001.', True),
    ('Friday October twenty-third at four thirty PM, B K zero zero zero one.', True),
    ('Friday October twenty three at four thirty PM, BK0001.', True),
    ('Friday October twenty four at four thirty PM, BK0001.', False),
    ('Friday October twenty second at four thirty PM, BK0001.', False),
    ('Friday November 23 at 4:30 PM, BK0001.', False),
    ('Friday at 4:30 PM, BK0001.', False),
])
def test_waveform_guard_requires_exact_explicit_calendar_date(text, expected):
    path = Path(__file__).resolve().parents[1] / 'scripts/scheduling_smoke.py'
    spec = importlib.util.spec_from_file_location('native_smoke_strict', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.native_fact_matches(text, '2026-10-23T16:30:00-05:00', 'BK-0001', strict_date=True) is expected


def test_reference_groups_preserve_nonzero_digits_and_reject_extra_zero():
    path = Path(__file__).resolve().parents[1] / 'scripts/scheduling_smoke.py'
    spec = importlib.util.spec_from_file_location('native_reference', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    text = 'Friday October 23 at four thirty PM, B K double zero one two.'
    assert m.native_fact_matches(text, '2026-10-23T16:30:00-05:00', 'BK-0012', strict_date=True)
    assert not m.native_fact_matches(text, '2026-10-23T16:30:00-05:00', 'BK-00012', strict_date=True)


@pytest.mark.parametrize("speech,reference,expected", [
    ("BK number one", "BK-0001", True),
    ("B K number 1", "BK-0001", True),
    ("B K number twelve", "BK-0012", True),
    ("BK1", "BK-0001", False),
    ("BK number 2", "BK-0001", False),
    ("BK number 1", "BK-00001", False),
    ("BK number 1, BK0002", "BK-0001", False),
])
def test_explicit_spoken_alias_is_reversible_and_never_implicitly_pads_literal_ids(speech, reference, expected):
    path = Path(__file__).resolve().parents[1] / "scripts/scheduling_smoke.py"
    spec = importlib.util.spec_from_file_location("native_alias", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    text = "Friday October 23 at 4:30 PM. Your booking reference is " + speech
    assert m.native_fact_matches(text, "2026-10-23T16:30:00-05:00", reference, strict_date=True) is expected


def test_probe_empty_glob_fails_before_loading_optional_asr_dependencies(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'experiments/scheduling/native-injection/transcribe_probe.py'
    result = subprocess.run([sys.executable,str(script),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode == 2 and 'No waveform files matched' in result.stderr
    assert not (tmp_path/'waveform-qwen-asr.json').exists()
