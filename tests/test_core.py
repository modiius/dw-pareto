import json
import subprocess

import pytest

from pareto import LedgerError, Tracker, fold_hash

METRICS = {"f1": "max", "latency_ms": "min"}


def tracker(tmp_path, **kw):
    kw.setdefault("metrics", METRICS)
    return Tracker(tmp_path / "exp", cwd=tmp_path, **kw)


def test_fold_hash_ignores_index_order_but_not_fold_order():
    a = [([1, 2, 3], [4, 5]), ([4, 5], [1, 2, 3])]
    b = [([3, 2, 1], [5, 4]), ([5, 4], [3, 2, 1])]
    assert fold_hash(a) == fold_hash(b)
    assert fold_hash(a) != fold_hash(list(reversed(a)))


def test_direction_is_validated(tmp_path):
    with pytest.raises(ValueError, match="max"):
        Tracker(tmp_path / "exp", metrics={"f1": "biggest"})


def test_missing_score_rejected(tmp_path):
    tr = tracker(tmp_path)
    with pytest.raises(ValueError, match="latency_ms"):
        tr.log({"f1": 0.9})


def test_settings_persist_and_reload(tmp_path):
    tracker(tmp_path, eps={"f1": 0.01}, alpha=0.1)
    again = Tracker(tmp_path / "exp")
    assert again.metrics == METRICS
    assert again.eps["f1"] == 0.01
    assert again.alpha == 0.1
    assert json.loads((tmp_path / "exp" / "config.json").read_text())["version"] == 1


def test_scalar_domination(tmp_path):
    tr = tracker(tmp_path)
    good = tr.log({"f1": 0.9, "latency_ms": 1.0})
    bad = tr.log({"f1": 0.8, "latency_ms": 2.0})
    assert good["on_front"] and not bad["on_front"]
    assert [r["run_id"] for r in tr.front()] == [good["run_id"]]


def test_trade_off_keeps_both(tmp_path):
    tr = tracker(tmp_path)
    tr.log({"f1": 0.9, "latency_ms": 9.0})
    tr.log({"f1": 0.8, "latency_ms": 1.0})
    assert len(tr.front()) == 2


def test_eps_swallows_a_tiny_gain(tmp_path):
    tr = tracker(tmp_path, eps={"f1": 0.01, "latency_ms": 0.5})
    first = tr.log({"f1": 0.900, "latency_ms": 1.0})
    tr.log({"f1": 0.905, "latency_ms": 1.1})
    assert [r["run_id"] for r in tr.front()] == [first["run_id"]]


def test_context_isolates_runs(tmp_path):
    tr = tracker(tmp_path)
    tr.log({"f1": 0.9, "latency_ms": 1.0}, context={"hw": "a100"})
    tr.log({"f1": 0.5, "latency_ms": 9.0}, context={"hw": "cpu"})
    assert len(tr.front()) == 2          # never compared across hardware
    assert len(tr.front(ignore=("hw",))) == 1
    assert len(tr.groups()) == 2


def test_per_fold_noise_is_not_an_improvement(tmp_path):
    tr = tracker(tmp_path)
    kw = dict(n_train=4000, n_test=1000, context={"folds": "abc"})
    a = [0.80, 0.84, 0.76, 0.88, 0.72]
    b = [0.81, 0.83, 0.77, 0.87, 0.73]   # +0.01 on average, swamped by fold spread
    tr.log({"f1": a, "latency_ms": 1.0}, **kw)
    tr.log({"f1": b, "latency_ms": 1.0}, **kw)
    assert len(tr.front()) == 1          # second is indistinguishable, earlier one wins


def test_per_fold_consistent_gain_dominates(tmp_path):
    tr = tracker(tmp_path)
    kw = dict(n_train=4000, n_test=1000, context={"folds": "abc"})
    tr.log({"f1": [0.70, 0.71, 0.69, 0.72, 0.68], "latency_ms": 1.0}, **kw)
    win = tr.log({"f1": [0.90, 0.91, 0.89, 0.92, 0.88], "latency_ms": 1.0}, **kw)
    assert win["on_front"]
    assert [r["run_id"] for r in tr.front()] == [win["run_id"]]


def test_mismatched_fold_counts_in_one_context_is_an_error(tmp_path):
    tr = tracker(tmp_path)
    tr.log({"f1": [0.7, 0.8], "latency_ms": 1.0}, n_train=10, n_test=5, context={"folds": "abc"})
    # the row is written; the front cannot be computed until the contexts are fixed
    with pytest.raises(LedgerError, match="fold counts differ"):
        tr.log({"f1": [0.7, 0.8, 0.9], "latency_ms": 1.0}, n_train=10, n_test=5,
               context={"folds": "abc"})
    assert len(tr.runs()) == 2
    with pytest.raises(LedgerError, match="fold counts differ"):
        tr.front()


def test_mismatched_split_sizes_in_one_context_is_an_error(tmp_path):
    tr = tracker(tmp_path)
    tr.log({"f1": [0.7, 0.8], "latency_ms": 1.0}, n_train=10, n_test=5, context={"folds": "abc"})
    with pytest.raises(LedgerError, match="n_train/n_test differ"):
        tr.log({"f1": [0.9, 0.9], "latency_ms": 1.0}, n_train=20, n_test=5,
               context={"folds": "abc"})


def test_per_fold_requires_split_sizes(tmp_path):
    tr = tracker(tmp_path)
    with pytest.raises(ValueError, match="n_train and n_test"):
        tr.log({"f1": [0.7, 0.8], "latency_ms": 1.0})


def test_ragged_score_lists_rejected(tmp_path):
    tr = tracker(tmp_path)
    with pytest.raises(ValueError, match="different lengths"):
        tr.log({"f1": [0.7, 0.8], "latency_ms": [1.0]}, n_train=10, n_test=5)


def test_string_score_is_not_a_fold_list(tmp_path):
    tr = tracker(tmp_path)
    tr.log({"f1": "0.9", "latency_ms": 1.0})              # coerced, not split into characters
    assert tr.runs()[-1]["scores"]["f1"] == 0.9
    with pytest.raises(ValueError):
        tr.log({"f1": "high", "latency_ms": 1.0})


def test_artifact_is_written_and_addressable(tmp_path):
    tr = tracker(tmp_path)
    res = tr.log({"f1": 0.9, "latency_ms": 1.0},
                 save=lambda d: (d / "model.txt").write_text("weights"))
    assert (tr.path(res["run_id"]) / "model.txt").read_text() == "weights"


def test_failed_save_leaves_no_run_dir_and_no_row(tmp_path):
    tr = tracker(tmp_path)

    def boom(d):
        raise RuntimeError("no disk")

    with pytest.raises(RuntimeError):
        tr.log({"f1": 0.9, "latency_ms": 1.0}, save=boom)
    assert tr.runs() == []
    assert not list((tmp_path / "exp" / "runs").iterdir())


def test_truncated_ledger_line_is_reported(tmp_path):
    tr = tracker(tmp_path)
    tr.log({"f1": 0.9, "latency_ms": 1.0})
    with open(tr.ledger, "a") as f:
        f.write('{"run_id": "broken"')
    with pytest.raises(LedgerError, match="truncated"):
        tr.runs()


def test_explain_reports_why_a_run_lost(tmp_path):
    tr = tracker(tmp_path)
    win = tr.log({"f1": 0.9, "latency_ms": 1.0})
    lose = tr.log({"f1": 0.8, "latency_ms": 2.0})
    ex = tr.explain()
    assert ex["runs"][win["run_id"]]["status"] == "front"
    assert ex["runs"][lose["run_id"]]["status"] == "dominated"
    assert ex["runs"][lose["run_id"]]["dominated_by"] == [win["run_id"]]


def test_front_can_be_recomputed_with_other_settings(tmp_path):
    tr = tracker(tmp_path)
    tr.log({"f1": 0.9, "latency_ms": 9.0})
    tr.log({"f1": 0.8, "latency_ms": 1.0})
    assert len(tr.front()) == 2
    assert len(tr.front(metrics={"f1": "max"})) == 1          # accuracy only
    assert len(tr.front(eps={"f1": 1.0, "latency_ms": 100})) == 1  # nothing is a real gain


def test_git_metadata_comes_from_the_given_cwd(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin"}
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=env)
    run("init", "-q")
    (repo / "a.txt").write_text("x")
    run("add", "a.txt")
    run("commit", "-qm", "init")
    tr = Tracker(tmp_path / "exp", metrics=METRICS, cwd=repo)
    tr.log({"f1": 0.9, "latency_ms": 1.0})
    row = tr.runs()[-1]
    assert row["commit"] and len(row["commit"]) == 40
    assert row["dirty"] is False


def test_writing_the_ledger_does_not_count_as_a_dirty_worktree(tmp_path):
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin"}
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=env)
    run("init", "-q")
    (repo / "train.py").write_text("x = 1\n")
    run("add", "train.py")
    run("commit", "-qm", "init")

    tr = Tracker(repo / "experiments", metrics=METRICS, cwd=repo)
    tr.log({"f1": 0.9, "latency_ms": 1.0})
    assert tr.runs()[-1]["dirty"] is False       # only the ledger changed

    (repo / "train.py").write_text("x = 2\n")    # a real code change still counts
    tr.log({"f1": 0.8, "latency_ms": 1.0})
    assert tr.runs()[-1]["dirty"] is True
