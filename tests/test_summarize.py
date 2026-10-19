import json
import urllib.error

import pytest

from speaker_lab import summarize


SEGMENTS = [{"id": "seg_0001", "start": 62.5, "end": 65, "speaker": "SPEAKER_00", "text_raw": "我们周五前完成验证。"}]


class FakeResponse:
    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, count):
        return self.body[:count]


def test_saves_chinese_summary_with_local_ollama_payload(monkeypatch, tmp_path):
    calls = []

    def fake_open(request, timeout):
        calls.append((request, timeout))
        return FakeResponse({"done": True, "message": {"content": "- 验证期限为周五。[seg_0001 00:01:02.50]"}})

    monkeypatch.setattr(summarize.urllib.request, "urlopen", fake_open)
    output = tmp_path / "summary.md"
    result = summarize.summarize_segments(SEGMENTS, {}, output)
    payload = json.loads(calls[0][0].data)
    assert result["text"] == output.read_text(encoding="utf-8")
    assert "SPEAKER_00" in payload["messages"][1]["content"]
    assert "00:01:02.50" in payload["messages"][1]["content"]
    assert payload["stream"] is False and payload["think"] is False
    assert payload["keep_alive"] == 0 and "tools" not in payload
    assert calls[0][0].full_url == "http://127.0.0.1:11434/api/chat"


def test_empty_transcript_does_not_contact_server(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("must not call Ollama")

    monkeypatch.setattr(summarize.urllib.request, "urlopen", unexpected)
    result = summarize.summarize_segments([], {}, tmp_path / "empty.md")
    assert "没有可用" in result["text"]
    assert result["warnings"]


@pytest.mark.parametrize("body", [b"not-json", [], {"message": {"content": 2}}, {"done": True, "message": {"content": ""}}, {"done": False, "message": {"content": "partial"}}, {"done": True, "done_reason": "length", "message": {"content": "partial"}}])
def test_rejects_invalid_or_incomplete_response_without_output(monkeypatch, tmp_path, body):
    monkeypatch.setattr(summarize.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse(body))
    output = tmp_path / "summary.md"
    with pytest.raises(summarize.SummaryError):
        summarize.summarize_segments(SEGMENTS, {}, output)
    assert not output.exists()


def test_network_failure_is_actionable(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(summarize.urllib.request, "urlopen", fail)
    with pytest.raises(summarize.SummaryError, match="ollama serve"):
        summarize.summarize_segments(SEGMENTS, {}, tmp_path / "notes.md")


def test_long_transcript_keeps_evidence_labels_and_reduces(monkeypatch, tmp_path):
    calls = []

    def chat(content, options, *, reduction=False):
        calls.append((content, reduction))
        return "- 要点。[seg_0001 00:01:02.50]"

    monkeypatch.setattr(summarize, "_chat", chat)
    segments = [{**SEGMENTS[0], "text_raw": "长内容" * 600}]
    summarize.summarize_segments(segments, {"chunk_chars": 512}, tmp_path / "notes.md")
    source_calls = [content for content, reduction in calls if not reduction]
    assert len(source_calls) > 1
    assert all("seg_0001" in content and "00:01:02.50" in content for content in source_calls)
    assert all(len(content) <= 512 for content in source_calls)
    assert any(reduction for _, reduction in calls)


def test_rejects_oversized_server_response(monkeypatch, tmp_path):
    monkeypatch.setattr(summarize.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse(b"x" * 1_000_001))
    with pytest.raises(summarize.SummaryError, match="响应过大"):
        summarize.summarize_segments(SEGMENTS, {}, tmp_path / "notes.md")


def test_non_compressing_reduction_is_bounded(monkeypatch, tmp_path):
    calls = []

    def chat(content, options, *, reduction=False):
        calls.append(reduction)
        return "保留内容" * 200

    monkeypatch.setattr(summarize, "_chat", chat)
    result = summarize.summarize_segments([{**SEGMENTS[0], "text_raw": "文字" * 500}], {"chunk_chars": 512}, tmp_path / "notes.md")
    assert len(calls) < 20
    assert any("按分块保留" in warning for warning in result["warnings"])


def test_missing_citation_is_explicit_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(summarize, "_chat", lambda *args, **kwargs: "- 讨论了验证计划。")
    output = tmp_path / "summary.md"
    result = summarize.summarize_segments(SEGMENTS, {}, output)
    assert output.exists()
    assert any("未包含片段 ID 引用" in warning for warning in result["warnings"])


def test_unknown_citation_is_rejected_before_saving(monkeypatch, tmp_path):
    monkeypatch.setattr(summarize, "_chat", lambda *args, **kwargs: "- 验证计划。[seg_99999 00:01:02.50]")
    output = tmp_path / "summary.md"
    with pytest.raises(summarize.SummaryError, match="不存在的片段 ID：seg_99999"):
        summarize.summarize_segments(SEGMENTS, {}, output)
    assert not output.exists()


def test_fallback_segment_id_is_valid_in_final_reduction(monkeypatch, tmp_path):
    calls = []

    def chat(content, options, *, reduction=False):
        calls.append((content, reduction))
        return "- 验证计划。[seg_00001 00:01:02.50]"

    monkeypatch.setattr(summarize, "_chat", chat)
    segment = {key: value for key, value in SEGMENTS[0].items() if key != "id"}
    segment["text_raw"] = "较长的讨论内容" * 200
    result = summarize.summarize_segments([segment], {"chunk_chars": 512}, tmp_path / "summary.md")
    assert any(reduction for _, reduction in calls)
    assert all("seg_00001" in content for content, _ in calls)
    assert "seg_00001" in result["text"]
    assert not any("未包含片段 ID 引用" in warning for warning in result["warnings"])


def test_unknown_id_introduced_by_reduction_is_rejected(monkeypatch, tmp_path):
    def chat(content, options, *, reduction=False):
        return "- 纪要。[seg_99999 00:00:00.00]" if reduction else "- 要点。[seg_0001 00:01:02.50]"

    monkeypatch.setattr(summarize, "_chat", chat)
    output = tmp_path / "summary.md"
    with pytest.raises(summarize.SummaryError, match="不存在的片段 ID"):
        summarize.summarize_segments([{**SEGMENTS[0], "text_raw": "长内容" * 300}], {"chunk_chars": 512}, output)
    assert not output.exists()


@pytest.mark.parametrize("options", [{"url": "file:///tmp/data"}, {"num_ctx": False}, {"chunk_chars": float("nan")}, {"timeout_seconds": 0}])
def test_rejects_bad_configuration(monkeypatch, tmp_path, options):
    monkeypatch.setattr(summarize.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse({"done": True, "message": {"content": "ok"}}))
    with pytest.raises(summarize.SummaryError):
        summarize.summarize_segments(SEGMENTS, options, tmp_path / "notes.md")
