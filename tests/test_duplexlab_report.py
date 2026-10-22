from test_duplexlab_metrics import result_case


def test_private_report_escapes_transcript_and_public_report_omits_originals(tmp_path):
    from speaker_lab.duplex.report import render_report
    _, _, run, _ = result_case(tmp_path, ["source secret"], ['<script>alert("secret")</script>'])
    private = render_report(run, tmp_path / "private.html").read_text()
    assert '<script>alert("secret")</script>' not in private
    assert '&lt;script&gt;' in private
    assert '<audio' in private
    public = render_report(run, tmp_path / "public.html", public=True).read_text()
    assert 'source secret' not in public and 'alert' not in public
    assert str(tmp_path) not in public and '<audio' not in public
