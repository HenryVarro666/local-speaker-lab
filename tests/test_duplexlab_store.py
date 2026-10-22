import pytest


def test_success_is_durable_and_cannot_be_run_twice(tmp_path):
    from speaker_lab.duplex.store import RunStore
    store = RunStore(tmp_path / "state.sqlite")
    attempt = store.start_attempt("clip", "baseline")
    store.finish_attempt("clip", "baseline", attempt, {"hypothesis": "hello"})
    reopened = RunStore(tmp_path / "state.sqlite")
    assert reopened.iter_results()[0]["hypothesis"] == "hello"
    with pytest.raises(ValueError, match="succeeded"):
        reopened.start_attempt("clip", "baseline")


def test_failed_attempt_can_retry_but_stale_attempt_cannot_commit(tmp_path):
    from speaker_lab.duplex.store import RunStore
    store = RunStore(tmp_path / "state.sqlite")
    first = store.start_attempt("clip", "baseline")
    store.fail_attempt("clip", "baseline", first, "model failed")
    second = store.start_attempt("clip", "baseline")
    assert second == first + 1
    with pytest.raises(ValueError, match="attempt"):
        store.finish_attempt("clip", "baseline", first, {"hypothesis": "stale"})
    assert store.iter_results() == []
    store.finish_attempt("clip", "baseline", second, {"hypothesis": "current"})
    assert store.iter_results()[0]["hypothesis"] == "current"


def test_running_job_and_nonfinite_result_do_not_become_success(tmp_path):
    from speaker_lab.duplex.store import RunStore
    store = RunStore(tmp_path / "state.sqlite")
    attempt = store.start_attempt("clip", "base")
    with pytest.raises(ValueError, match="running"):
        store.start_attempt("clip", "base")
    with pytest.raises(ValueError):
        store.finish_attempt("clip", "base", attempt, {"hypothesis": "bad", "seconds": float("nan")})
    assert store.iter_results() == []
