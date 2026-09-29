"""click CLI over the Tracker.

Every command is a thin shell around `pareto.core`: the ledger is the only state,
and the front is recomputed on every invocation.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import webbrowser
from pathlib import Path

import click

from pareto.core import CONFIG_NAME, DIRECTIONS, LedgerError, Tracker
from pareto.report import render

# ---------------------------------------------------------------- parsing


def _kv(value: str, sep: str, what: str) -> tuple[str, str]:
    if sep not in value:
        raise click.BadParameter(f"expected {what}, got {value!r}")
    k, v = value.split(sep, 1)
    k, v = k.strip(), v.strip()
    if not k:
        raise click.BadParameter(f"empty name in {value!r}")
    return k, v


def parse_metrics(pairs: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for p in pairs:
        name, direction = _kv(p, ":", "NAME:max or NAME:min")
        if direction not in DIRECTIONS:
            raise click.BadParameter(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        out[name] = direction
    return out


def parse_floats(pairs: tuple[str, ...]) -> dict[str, float]:
    out = {}
    for p in pairs:
        name, val = _kv(p, "=", "NAME=VALUE")
        try:
            out[name] = float(val)
        except ValueError:
            raise click.BadParameter(f"{name}: {val!r} is not a number") from None
    return out


def parse_scores(pairs: tuple[str, ...]) -> dict[str, float | list[float]]:
    """NAME=0.93 -> scalar; NAME=0.93,0.91,0.95 -> per-fold list."""
    out: dict[str, float | list[float]] = {}
    for p in pairs:
        name, val = _kv(p, "=", "NAME=VALUE or NAME=V1,V2,...")
        parts = [v for v in val.split(",") if v.strip()]
        try:
            nums = [float(v) for v in parts]
        except ValueError:
            raise click.BadParameter(f"{name}: {val!r} is not a number or comma-separated list") from None
        if not nums:
            raise click.BadParameter(f"{name}: no value given")
        out[name] = nums[0] if len(nums) == 1 and "," not in val else nums
    return out


def parse_context(pairs: tuple[str, ...]) -> dict[str, str]:
    return dict(_kv(p, "=", "KEY=VALUE") for p in pairs)


def load_json(path: str | None) -> dict:
    if not path:
        return {}
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise click.BadParameter(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------- display


def mean(v) -> float:
    return sum(v) / len(v) if isinstance(v, list) else v


def fmt(v) -> str:
    m = mean(v)
    s = f"{m:.6g}"
    return f"{s} ({len(v)}f)" if isinstance(v, list) else s


def pad(cell: str, width: int) -> str:
    """ljust that ignores ANSI styling, which is not printable width."""
    return str(cell) + " " * max(0, width - len(click.unstyle(str(cell))))


def describe(v, keep: int = 6) -> str:
    """Mean, spread and a glimpse of the folds - the full list is in --json."""
    if not isinstance(v, list):
        return fmt(v)
    m = mean(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    head = ", ".join(f"{x:.4g}" for x in v[:keep])
    more = f", \u2026 (+{len(v) - keep})" if len(v) > keep else ""
    return f"{m:.6g} \u00b1 {sd:.3g}  {len(v)} folds  [{min(v):.4g}, {max(v):.4g}]  [{head}{more}]"


def table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(click.unstyle(str(r[i]))) for r in [headers, *rows])
              for i in range(len(headers))]
    line = lambda cells: "  ".join(pad(c, w) for c, w in zip(cells, widths)).rstrip()
    out = [line([click.style(h, bold=True) for h in headers]),
           click.style(line(["-" * w for w in widths]), dim=True)]
    out += [line(r) for r in rows]
    return "\n".join(out)


STATUS_STYLE = {"front": {"fg": "green", "bold": True}, "tied": {"fg": "yellow"},
                "dominated": {"dim": True}}


def short_context(row: dict, limit: int = 40) -> str:
    ctx = ",".join(f"{k}={v}" for k, v in sorted(row["context"].items()))
    return ctx if len(ctx) <= limit else ctx[: limit - 1] + "…"


# ---------------------------------------------------------------- commands


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--root", "-C", default=".", envvar="PARETO_ROOT", show_default=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Tracker directory (ledger + artifacts). Env: PARETO_ROOT.")
@click.version_option(package_name="pareto")
@click.pass_context
def cli(ctx: click.Context, root: Path) -> None:
    """Track model runs and keep a Pareto front of the best ones.

    The ledger is append-only; the front is recomputed from it on every command,
    so nothing can drift out of sync.
    """
    ctx.obj = {"root": root}


def tracker(ctx: click.Context, **overrides) -> Tracker:
    root = ctx.obj["root"]
    if not (root / CONFIG_NAME).exists() and not overrides.get("metrics"):
        raise click.ClickException(f"no {CONFIG_NAME} in {root}; run `pareto -C {root} init ...` first")
    try:
        return Tracker(root, **overrides)
    except (ValueError, LedgerError) as e:
        raise click.ClickException(str(e)) from e


def guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except (ValueError, LedgerError, KeyError) as e:
        raise click.ClickException(str(e)) from e


metric_opt = click.option("--metric", "-m", "metric_specs", multiple=True, metavar="NAME:max|min",
                          help="Metric and its direction; repeat per metric.")
eps_opt = click.option("--eps", "eps_specs", multiple=True, metavar="NAME=VALUE",
                       help="Smallest difference worth caring about; repeat per metric.")
alpha_opt = click.option("--alpha", type=float, help="One-sided significance level [default: 0.05].")
ignore_opt = click.option("--ignore", multiple=True, metavar="KEY",
                          help="Context key to disregard when deciding comparability.")


@cli.command()
@metric_opt
@eps_opt
@alpha_opt
@click.pass_context
def init(ctx: click.Context, metric_specs, eps_specs, alpha) -> None:
    """Create the tracker directory and record its metrics.

    \b
    pareto -C experiments init -m f1:max -m latency_ms:min --eps f1=0.005
    """
    metrics = parse_metrics(metric_specs)
    if not metrics:
        raise click.UsageError("give at least one --metric NAME:max|min")
    tr = guard(Tracker, ctx.obj["root"], metrics=metrics, alpha=alpha, eps=parse_floats(eps_specs))
    click.echo(f"initialised {tr.root}")
    click.echo(json.dumps(tr.settings, indent=2))


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Show the recorded metrics, alpha and eps."""
    click.echo(json.dumps(tracker(ctx).settings, indent=2))


@cli.command()
@click.option("--score", "-s", "score_specs", multiple=True, metavar="NAME=VALUE[,VALUE...]",
              help="Metric value, or comma-separated per-fold values.")
@click.option("--scores-json", metavar="FILE", help="JSON object of scores ('-' for stdin).")
@click.option("--n-train", type=int, help="Training set size (required for per-fold scores).")
@click.option("--n-test", type=int, help="Test set size (required for per-fold scores).")
@click.option("--context", "-c", "context_specs", multiple=True, metavar="KEY=VALUE",
              help="What must match for two runs to be comparable (folds, hardware, ...).")
@click.option("--config-json", metavar="FILE", help="JSON object of hyper-parameters ('-' for stdin).")
@click.option("--artifact", type=click.Path(exists=True, path_type=Path),
              help="File or directory to copy into the run's artifact dir.")
@click.option("--note", help="Free-text label for this run.")
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@click.pass_context
def log(ctx, score_specs, scores_json, n_train, n_test, context_specs, config_json,
        artifact, note, as_json) -> None:
    """Record a run and report whether it landed on the front.

    \b
    pareto -C experiments log -s f1=0.91,0.93,0.90 -s latency_ms=3.2 \\
        --n-train 4000 --n-test 1000 -c folds=8f3a -c hw=a100 --artifact model.joblib
    """
    scores = load_json(scores_json) | parse_scores(score_specs)
    if not scores:
        raise click.UsageError("give --score or --scores-json")
    tr = tracker(ctx)

    save = None
    if artifact is not None:
        src = artifact.resolve()

        def save(d: Path) -> None:
            if src.is_dir():
                shutil.copytree(src, d / src.name)
            else:
                shutil.copy2(src, d / src.name)

    res = guard(tr.log, scores, n_train=n_train, n_test=n_test,
                context=parse_context(context_specs), config=load_json(config_json),
                note=note, save=save)
    if as_json:
        click.echo(json.dumps(res))
        return
    mark = click.style("on front", fg="green", bold=True) if res["on_front"] else click.style(
        "not on front", dim=True)
    click.echo(f"{res['run_id']}  {mark}")
    if res["artifact"]:
        click.echo(f"artifact: {tr.path(res['run_id'])}")


@cli.command()
@metric_opt
@eps_opt
@alpha_opt
@ignore_opt
@click.option("--json", "as_json", is_flag=True, help="Print the front rows as JSON.")
@click.pass_context
def front(ctx, metric_specs, eps_specs, alpha, ignore, as_json) -> None:
    """Show the current Pareto front.

    Metrics, eps and alpha default to the recorded settings; overriding them here
    asks a different question of the same ledger without changing anything.
    """
    tr = tracker(ctx)
    metrics = parse_metrics(metric_specs) or None
    rows = guard(tr.front, metrics=metrics, alpha=alpha, eps=parse_floats(eps_specs), ignore=ignore)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("no runs yet")
        return
    ms = list(metrics or tr.metrics)
    body = [[r["run_id"], *[fmt(r["scores"][m]) for m in ms], short_context(r), r.get("note") or ""]
            for r in rows]
    click.echo(table(body, ["run_id", *ms, "context", "note"]))
    groups = guard(tr.groups, ignore)
    if len(groups) > 1:
        click.echo(click.style(
            f"\nnote: {len(groups)} context groups; runs in different groups are never compared. "
            "Use --ignore KEY to merge them.", fg="yellow"))


@cli.command()
@metric_opt
@eps_opt
@alpha_opt
@ignore_opt
@click.option("--all", "show_all", is_flag=True, help="Include dominated runs.")
@click.option("--json", "as_json", is_flag=True, help="Print the rows as JSON.")
@click.pass_context
def runs(ctx, metric_specs, eps_specs, alpha, ignore, show_all, as_json) -> None:
    """List runs with their status (front / tied / dominated)."""
    tr = tracker(ctx)
    metrics = parse_metrics(metric_specs) or None
    rows = guard(tr.runs)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("no runs yet")
        return
    ex = guard(tr.explain, metrics=metrics, alpha=alpha, eps=parse_floats(eps_specs), ignore=ignore)
    ms = list(metrics or tr.metrics)
    body = []
    for r in rows:
        info = ex["runs"].get(r["run_id"])
        status = info["status"] if info else "n/a"
        if status == "dominated" and not show_all:
            continue
        body.append([r["run_id"], click.style(status, **STATUS_STYLE.get(status, {})),
                     *[fmt(r["scores"][m]) if m in r["scores"] else "-" for m in ms],
                     short_context(r), r.get("note") or ""])
    click.echo(table(body, ["run_id", "status", *ms, "context", "note"]))
    hidden = len(rows) - len(body)
    if hidden:
        click.echo(click.style(f"\n{hidden} dominated run(s) hidden; --all to show.", dim=True))


@cli.command()
@click.argument("run_id")
@ignore_opt
@click.option("--json", "as_json", is_flag=True, help="Print the ledger row as JSON.")
@click.pass_context
def show(ctx, run_id, ignore, as_json) -> None:
    """Show everything recorded about one run."""
    tr = tracker(ctx)
    row = guard(tr.get, run_id)
    if as_json:
        click.echo(json.dumps(row, indent=2))
        return
    ex = guard(tr.explain, ignore=ignore)["runs"].get(run_id, {})
    out = [("run_id", row["run_id"]),
           ("created", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["created_at"]))),
           ("status", click.style(ex.get("status", "n/a"), **STATUS_STYLE.get(ex.get("status"), {})))]
    if ex.get("dominated_by"):
        out.append(("dominated by", ", ".join(ex["dominated_by"])))
    if ex.get("tied_with"):
        out.append(("tied with", ", ".join(ex["tied_with"])))
    for m, v in row["scores"].items():
        out.append((f"score {m}", describe(v)))
    out += [("n_train", row["n_train"]), ("n_test", row["n_test"]),
            ("commit", f"{row.get('commit') or '-'}{' (dirty)' if row.get('dirty') else ''}"),
            ("note", row.get("note") or "-"),
            ("artifact", str(tr.path(run_id)) if row["artifact"] else "-")]
    width = max(len(k) for k, _ in out)
    for k, v in out:
        click.echo(f"{click.style(k.ljust(width), bold=True)}  {v}")
    for label, d in (("context", row["context"]), ("config", row["config"]), ("env", row["env"])):
        if d:
            click.echo(f"\n{click.style(label, bold=True)}\n" + json.dumps(d, indent=2))


@cli.command()
@click.argument("run_id")
@click.pass_context
def path(ctx, run_id) -> None:
    """Print the artifact directory of a run."""
    tr = tracker(ctx)
    row = guard(tr.get, run_id)
    if not row["artifact"]:
        raise click.ClickException(f"{run_id} has no artifact")
    click.echo(tr.path(run_id))


@cli.command()
@click.option("--out", "-o", default="pareto.html", show_default=True,
              type=click.Path(dir_okay=False, path_type=Path), help="Where to write the report.")
@metric_opt
@eps_opt
@alpha_opt
@ignore_opt
@click.option("--open", "open_", is_flag=True, help="Open the report in a browser.")
@click.option("--title", help="Report title [default: the root directory name].")
@click.option("--embed", is_flag=True,
              help="Omit the <html>/<head>/<body> wrapper, for embedding or publishing.")
@click.pass_context
def report(ctx, out, metric_specs, eps_specs, alpha, ignore, open_, title, embed) -> None:
    """Write a self-contained HTML view of the front.

    \b
    pareto -C experiments report -o front.html --open
    """
    tr = tracker(ctx)
    metrics = parse_metrics(metric_specs) or None
    rows = guard(tr.runs)
    if not rows:
        raise click.ClickException("no runs yet")
    ex = guard(tr.explain, metrics=metrics, alpha=alpha, eps=parse_floats(eps_specs), ignore=ignore)
    html = render(rows, ex, title=title or tr.root.resolve().name,
                  root=str(tr.root.resolve()), embed=embed)
    out.write_text(html, encoding="utf-8")
    click.echo(f"wrote {out}  ({len(rows)} runs, "
               f"{sum(1 for v in ex['runs'].values() if v['status'] == 'front')} on front)")
    if open_:
        webbrowser.open(out.resolve().as_uri())
