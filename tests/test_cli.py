import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pareto.cli import cli, parse_scores


@pytest.fixture
def run(tmp_path):
    runner = CliRunner()

    def invoke(*args, expect=0):
        res = runner.invoke(cli, ["-C", str(tmp_path / "exp"), *args], catch_exceptions=False)
        assert res.exit_code == expect, res.output
        return res

    return invoke


def init(run):
    return run("init", "-m", "f1:max", "-m", "latency_ms:min", "--eps", "f1=0.005")


def test_parse_scores_scalar_vs_folds():
    assert parse_scores(("f1=0.9",)) == {"f1": 0.9}
    assert parse_scores(("f1=0.9,0.8",)) == {"f1": [0.9, 0.8]}
    assert parse_scores(("f1=0.9,",)) == {"f1": [0.9]}


def test_commands_need_an_initialised_root(run):
    res = run("front", expect=1)
    assert "pareto" in res.output and "init" in res.output


def test_init_records_settings(run, tmp_path):
    init(run)
    cfg = json.loads((tmp_path / "exp" / "config.json").read_text())
    assert cfg["metrics"] == {"f1": "max", "latency_ms": "min"}
    assert cfg["eps"]["f1"] == 0.005


def test_bad_direction_is_rejected(run):
    res = run("init", "-m", "f1:up", expect=2)
    assert "direction" in res.output


def test_log_front_and_show(run):
    init(run)
    a = json.loads(run("log", "-s", "f1=0.90", "-s", "latency_ms=1.0", "--json").output)
    b = json.loads(run("log", "-s", "f1=0.80", "-s", "latency_ms=2.0", "--json").output)
    assert a["on_front"] and not b["on_front"]

    front = run("front").output
    assert a["run_id"] in front and b["run_id"] not in front

    runs = run("runs", "--all").output
    assert "dominated" in runs and b["run_id"] in runs

    detail = run("show", b["run_id"]).output
    assert a["run_id"] in detail   # reported as the run that dominates it


def test_log_with_folds_and_context(run):
    init(run)
    res = run("log", "-s", "f1=0.90,0.91,0.89", "-s", "latency_ms=1.0",
              "--n-train", "400", "--n-test", "100", "-c", "folds=abc", "--note", "baseline")
    assert "on front" in res.output
    assert "baseline" in run("front").output


def test_log_rejects_folds_without_split_sizes(run):
    init(run)
    res = run("log", "-s", "f1=0.9,0.8", "-s", "latency_ms=1.0", expect=1)
    assert "n_train" in res.output


def test_scores_and_config_from_json(run, tmp_path):
    init(run)
    scores = tmp_path / "s.json"
    scores.write_text(json.dumps({"f1": [0.9, 0.92], "latency_ms": 2.0}))
    params = tmp_path / "c.json"
    params.write_text(json.dumps({"depth": 7}))
    out = run("log", "--scores-json", str(scores), "--config-json", str(params),
              "--n-train", "100", "--n-test", "20", "--json").output
    run_id = json.loads(out)["run_id"]
    assert '"depth": 7' in run("show", run_id, "--json").output


def test_artifact_is_copied(run, tmp_path):
    init(run)
    model = tmp_path / "model.txt"
    model.write_text("weights")
    out = run("log", "-s", "f1=0.9", "-s", "latency_ms=1.0", "--artifact", str(model), "--json").output
    run_id = json.loads(out)["run_id"]
    where = Path(run("path", run_id).output.strip())
    assert (where / "model.txt").read_text() == "weights"


def test_front_can_be_recomputed_with_other_metrics(run):
    init(run)
    run("log", "-s", "f1=0.90", "-s", "latency_ms=9.0")
    run("log", "-s", "f1=0.80", "-s", "latency_ms=1.0")
    assert len(json.loads(run("front", "--json").output)) == 2
    assert len(json.loads(run("front", "-m", "f1:max", "--json").output)) == 1


def test_context_groups_are_flagged(run):
    init(run)
    run("log", "-s", "f1=0.9", "-s", "latency_ms=1.0", "-c", "hw=a100")
    run("log", "-s", "f1=0.5", "-s", "latency_ms=9.0", "-c", "hw=cpu")
    assert "context groups" in run("front").output
    assert "context groups" not in run("front", "--ignore", "hw").output


def test_report_is_self_contained(run, tmp_path):
    init(run)
    out = json.loads(run("log", "-s", "f1=0.9", "-s", "latency_ms=1.0", "--json").output)
    html = tmp_path / "front.html"
    run("report", "-o", str(html))
    text = html.read_text()
    assert out["run_id"] in text
    assert "__PARETO_DATA__" not in text
    assert "fetch(" not in text                    # nothing is loaded at view time
    assert "<script src" not in text and "<link rel=\"stylesheet\"" not in text


def test_report_needs_runs(run):
    init(run)
    res = run("report", expect=1)
    assert "no runs" in res.output


def test_report_title_reaches_the_document(run, tmp_path):
    init(run)
    run("log", "-s", "f1=0.9", "-s", "latency_ms=1.0")
    html = tmp_path / "t.html"
    run("report", "-o", str(html), "--title", "Sweep <one>")
    assert "<title>Sweep &lt;one&gt;</title>" in html.read_text()


def test_embed_drops_the_document_wrapper(run, tmp_path):
    init(run)
    out = json.loads(run("log", "-s", "f1=0.9", "-s", "latency_ms=1.0", "--json").output)
    full, embed = tmp_path / "a.html", tmp_path / "b.html"
    run("report", "-o", str(full))
    run("report", "-o", str(embed), "--embed")
    text = embed.read_text()
    for tag in ("<!DOCTYPE", "<html", "<head>", "<body>"):
        assert tag not in text
    assert text.startswith("<title>")
    assert "<style>" in text and out["run_id"] in text
    assert len(text) < len(full.read_text())
