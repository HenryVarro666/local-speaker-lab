import subprocess
import sys

from test_duplexlab_metrics import result_case


def test_help_does_not_import_torch_or_start_a_model():
    code = """
import sys
from speaker_lab.duplex.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules and 'qwen_asr' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_cli_scores_and_builds_a_replay_from_a_complete_run(tmp_path):
    root, protocol, run, _ = result_case(tmp_path)
    output = tmp_path / "scores.json"
    completed = subprocess.run([sys.executable, "-m", "speaker_lab.duplex", "score", "--run", str(run),
                                "--protocol", str(protocol), "--data-root", str(root), "--output", str(output)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    report = tmp_path / "index.html"
    completed = subprocess.run([sys.executable, "-m", "speaker_lab.duplex", "report", "--run", str(run),
                                "--output", str(report)], capture_output=True, text=True)
    assert completed.returncode == 0 and "<audio" in report.read_text(), completed.stderr
