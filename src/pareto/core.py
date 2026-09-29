"""core.py - a lightweight Pareto-front tracker for model evaluations.

One append-only JSONL ledger of runs. The front is recomputed from the ledger on
every call, so there is no stored front to drift and no incremental logic to get
wrong.

    from pareto import Tracker, fold_hash

    folds = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=0).split(X))
    tr = Tracker("experiments", metrics={"f1": "max", "latency_ms": "min"},
                 eps={"f1": 0.005, "latency_ms": 0.5})

    res = tr.log({"f1": per_fold_f1, "latency_ms": 3.2},
                 n_train=len(folds[0][0]), n_test=len(folds[0][1]),
                 context={"folds": fold_hash(folds), "hw": "a100/bs32/8thr"},
                 config=params,
                 save=lambda d: joblib.dump(model, d / "model.joblib"))
    res["on_front"]          # True / False
    tr.front()               # list of ledger rows, recomputed
    tr.path(run_id)          # artifact dir; loading is your code's job

Restoring code: `git worktree add ../restore <row["commit"]>` - the commit SHA is
recorded.

Settings (metrics, alpha, eps) are persisted to `<root>/config.json` so that the
CLI - and a later session - recompute the same front as the script that wrote the
ledger. They are inputs to a pure recomputation, never baked into the rows, so
changing them is safe and retroactive.

Caveats that are inherent to the method, not bugs:

* `front()` is a screening device, not an inferential claim. Every pair of runs
  is tested on every metric with no multiplicity correction, so the family-wise
  error rate is far above `alpha`. Treat front membership as "not yet ruled out".
* "Indistinguishable" is not transitive: A~B and B~C does not give A~C. Step 2 of
  `front()` (dropping runs equivalent to one already kept) therefore depends on
  ledger order, which is time order. Earlier runs win ties - deliberately, so the
  front does not churn when a re-run lands on the same numbers.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from scipy import stats

CONFIG_NAME = "config.json"
LEDGER_NAME = "ledger.jsonl"
LOCKFILES = ("uv.lock", "poetry.lock", "pixi.lock", "requirements.lock", "requirements.txt", "environment.yml")
DIRECTIONS = ("max", "min")


class LedgerError(RuntimeError):
    """The ledger on disk is unreadable or internally inconsistent."""


def fold_hash(splits: Iterable[tuple[Sequence[int], Sequence[int]]]) -> str:
    """Hash of CV fold indices (fold order matters; index order within a fold doesn't)."""
    h = hashlib.sha256()
    for train, test in splits:
        for idx in (train, test):
            h.update(",".join(map(str, sorted(int(i) for i in idx))).encode())
            h.update(b"|")
        h.update(b";")
    return h.hexdigest()[:16]


def _git(cwd: Path, exclude: Path | None = None) -> dict:
    """Commit of `cwd`'s repo, and whether the worktree was dirty when the run happened.

    `exclude` (the tracker root) is left out of the dirty check: appending to your
    own ledger is not a change to the code under evaluation, and without this every
    row in a repo that tracks its ledger would be flagged.
    """
    try:
        def run(*a: str) -> str:
            return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True,
                                  check=True).stdout.strip()
        status = ["status", "--porcelain"]
        if exclude is not None:
            try:
                rel = exclude.resolve().relative_to(Path(run("rev-parse", "--show-toplevel")).resolve())
                status += ["--", ".", f":(exclude){rel.as_posix()}"]
            except ValueError:
                pass  # the tracker lives outside the repo; nothing to exclude
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run(*status))}
    except Exception:
        return {"commit": None, "dirty": None}


def _env(cwd: Path) -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for name in LOCKFILES:
        p = cwd / name
        if p.is_file():
            env["lockfile"] = name
            env["lockfile_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            break
    return env


@lru_cache(maxsize=None)
def _t_crit(alpha: float, df: int) -> float:
    return float(stats.t.ppf(1 - alpha, df))


def _is_seq(v: Any) -> bool:
    """True for a per-fold sequence. Strings and mappings are not per-fold scores."""
    return not isinstance(v, (str, bytes, Mapping)) and hasattr(v, "__len__")


def _mean(v: float | list[float]) -> float:
    return sum(v) / len(v) if isinstance(v, list) else v


class Tracker:
    """Append-only run ledger with a recomputed Pareto front.

    metrics: {name: "max" | "min"}. eps: minimum practically relevant difference
    per metric - a difference smaller than eps is treated as no difference, which
    is what keeps the front from filling up with noise.

    Arguments given explicitly are persisted to `<root>/config.json` and become
    the defaults for later sessions and for the CLI; omitted ones are loaded from
    there. Since the front is always recomputed, changing them is retroactive and
    safe.
    """

    def __init__(self, root: str | Path, metrics: Mapping[str, str] | None = None,
                 alpha: float | None = None, eps: Mapping[str, float] | None = None,
                 cwd: str | Path | None = None):
        self.root = Path(root)
        self.ledger = self.root / LEDGER_NAME
        self.config_path = self.root / CONFIG_NAME
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()

        stored = self._read_config()
        if metrics is None:
            metrics = stored.get("metrics")
        if metrics is None:
            raise ValueError(f"no metrics given and no {CONFIG_NAME} in {self.root}; "
                             "pass metrics={...} or run `pareto init`")
        for m, d in metrics.items():
            if d not in DIRECTIONS:
                raise ValueError(f"metric {m!r}: direction must be 'max' or 'min', got {d!r}")

        self.metrics = dict(metrics)
        self.alpha = stored.get("alpha", 0.05) if alpha is None else float(alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        self.eps = {m: 0.0 for m in self.metrics} | dict(stored.get("eps") or {}) | dict(eps or {})
        for m, e in self.eps.items():
            if e < 0:
                raise ValueError(f"eps[{m!r}] must be >= 0, got {e}")

        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self._write_config()

    # ---------- settings ----------
    def _read_config(self) -> dict:
        try:
            return json.loads(self.config_path.read_text())
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            raise LedgerError(f"{self.config_path} is not valid JSON: {e}") from e

    def _write_config(self) -> None:
        cfg = {"version": 1, "metrics": self.metrics, "alpha": self.alpha, "eps": self.eps}
        if self._read_config() == cfg:
            return
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, indent=2) + "\n")
        tmp.replace(self.config_path)

    @property
    def settings(self) -> dict:
        return {"metrics": dict(self.metrics), "alpha": self.alpha, "eps": dict(self.eps)}

    # ---------- writing ----------
    def log(self, scores: Mapping[str, float | Sequence[float]], *,
            n_train: int | None = None, n_test: int | None = None,
            context: Mapping | None = None, config: Mapping | None = None,
            note: str | None = None,
            save: Callable[[Path], None] | None = None) -> dict:
        """Record a run. Per-fold lists get the corrected t-test; scalars are compared by value +/- eps.

        The row is written before the front is recomputed, so a LedgerError raised
        here (contexts that claim to match but cannot be compared) means the run was
        recorded and the ledger needs fixing, not that the run was lost.
        """
        missing = set(self.metrics) - set(scores)
        if missing:
            raise ValueError(f"missing scores for {sorted(missing)}")
        clean = {m: ([float(x) for x in v] if _is_seq(v) else float(v)) for m, v in scores.items()}
        empty = [m for m, v in clean.items() if isinstance(v, list) and not v]
        if empty:
            raise ValueError(f"empty per-fold list for {empty}")
        lens = {len(v) for v in clean.values() if isinstance(v, list)}
        if len(lens) > 1:
            raise ValueError(f"per-fold lists have different lengths: {lens}")
        if lens and (not n_train or not n_test):
            raise ValueError("n_train and n_test are required for per-fold scores")

        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        artifact = None
        if save is not None:  # write to temp dir, then rename: a crash leaves only a .tmp dir
            tmp = self.root / "runs" / f".tmp-{run_id}"
            tmp.mkdir()
            try:
                save(tmp)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            tmp.rename(self.root / "runs" / run_id)
            artifact = f"runs/{run_id}"

        row = {"run_id": run_id, "created_at": time.time(), "scores": clean,
               "n_train": n_train, "n_test": n_test, "context": dict(context or {}),
               "config": dict(config or {}), "note": note, **_git(self.cwd, self.root),
               "env": _env(self.cwd), "artifact": artifact}
        with open(self.ledger, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        on_front = any(r["run_id"] == run_id for r in self.front())
        return {"run_id": run_id, "on_front": on_front, "artifact": artifact}

    # ---------- reading ----------
    def runs(self) -> list[dict]:
        if not self.ledger.exists():
            return []
        rows = []
        with open(self.ledger) as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # writers hold LOCK_EX; never read a half-written line
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise LedgerError(
                        f"{self.ledger}:{n} is not valid JSON ({e}). A crashed write can leave a "
                        "truncated last line; delete that line to recover."
                    ) from e
        return rows

    def get(self, run_id: str) -> dict:
        for r in self.runs():
            if r["run_id"] == run_id:
                return r
        raise KeyError(run_id)

    def path(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def front(self, metrics: Mapping[str, str] | None = None, alpha: float | None = None,
              eps: Mapping[str, float] | None = None, ignore: Iterable[str] = ()) -> list[dict]:
        """Recompute the front from the ledger. Pure function of (ledger, settings).

        ignore: context keys to disregard, e.g. ignore=("hw",) for an accuracy-only front
        across machines. Runs with any other context difference are never compared.
        """
        rows, cfg = self._prepare(metrics, alpha, eps, ignore)
        # 1) non-dominated runs
        nd = [r for r in rows if not any(s is not r and self._dominates(s, r, cfg) for s in rows)]
        # 2) among those, drop runs indistinguishable from an earlier kept run
        front: list[dict] = []
        for r in nd:  # ledger order == time order
            if not any(self._equivalent(f, r, cfg) for f in front):
                front.append(r)
        return front

    def explain(self, metrics: Mapping[str, str] | None = None, alpha: float | None = None,
                eps: Mapping[str, float] | None = None, ignore: Iterable[str] = ()) -> dict:
        """Everything the report needs: per-run status and who ruled each run out.

        status is "front", "dominated" (a listed run beats it) or "tied" (nothing
        beats it, but an earlier front member is indistinguishable from it).
        """
        rows, cfg = self._prepare(metrics, alpha, eps, ignore)
        front = {r["run_id"] for r in self.front(metrics, alpha, eps, ignore)}
        out = {}
        for r in rows:
            by = [s["run_id"] for s in rows if s is not r and self._dominates(s, r, cfg)]
            if r["run_id"] in front:
                status = "front"
            else:
                status = "dominated" if by else "tied"
            tied = [s["run_id"] for s in rows if s is not r and self._equivalent(s, r, cfg)]
            out[r["run_id"]] = {"status": status, "dominated_by": by, "tied_with": tied,
                                "group": self._group_key(r, cfg[3])}
        return {"runs": out, "metrics": cfg[0], "alpha": cfg[1], "eps": cfg[2],
                "ignore": sorted(cfg[3]), "n_compared": len(rows)}

    def groups(self, ignore: Iterable[str] = ()) -> dict[str, list[str]]:
        """Context groups. Runs in different groups are never compared to each other."""
        out: dict[str, list[str]] = {}
        for r in self.runs():
            out.setdefault(self._group_key(r, frozenset(ignore)), []).append(r["run_id"])
        return out

    def _prepare(self, metrics, alpha, eps, ignore) -> tuple[list[dict], tuple]:
        metrics = dict(metrics or self.metrics)
        for m, d in metrics.items():
            if d not in DIRECTIONS:
                raise ValueError(f"metric {m!r}: direction must be 'max' or 'min', got {d!r}")
        cfg = (metrics, self.alpha if alpha is None else alpha,
               self.eps | dict(eps or {}), frozenset(ignore))
        rows = [r for r in self.runs() if all(m in r["scores"] for m in metrics)]
        return rows, cfg

    # ---------- statistics ----------
    @staticmethod
    def _group_key(r, ignore: frozenset) -> str:
        ctx = {k: v for k, v in sorted(r["context"].items()) if k not in ignore}
        return json.dumps(ctx, sort_keys=True, default=str)

    @classmethod
    def _comparable(cls, a, b, cfg) -> bool:
        return cls._group_key(a, cfg[3]) == cls._group_key(b, cfg[3])

    @staticmethod
    def _better(a, b, m, cfg) -> bool:
        """a confidently better than b on metric m by more than eps."""
        metrics, alpha, eps, _ = cfg
        sign = 1.0 if metrics[m] == "max" else -1.0
        x, y, e = a["scores"][m], b["scores"][m], eps.get(m, 0.0)
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                raise LedgerError(f"{a['run_id']} vs {b['run_id']}: fold counts differ on {m!r} "
                                  f"({len(x)} vs {len(y)}) but their contexts match; "
                                  "put a fold hash in `context` so they are not compared")
            if (a["n_train"], a["n_test"]) != (b["n_train"], b["n_test"]):
                raise LedgerError(f"{a['run_id']} vs {b['run_id']}: n_train/n_test differ "
                                  f"({a['n_train']}/{a['n_test']} vs {b['n_train']}/{b['n_test']}) "
                                  "but their contexts match; put a fold hash in `context` "
                                  "so they are not compared")
            d = [sign * (xi - yi) for xi, yi in zip(x, y)]
            J = len(d)
            mean = sum(d) / J
            if mean <= e:
                return False
            if J == 1:
                return True  # no variance estimate from one fold: the eps rule is all there is
            var = sum((di - mean) ** 2 for di in d) / (J - 1)
            if var == 0.0:
                return True  # identical difference on every fold
            # Nadeau-Bengio corrected resampled t-test, one-sided, H0: true diff <= eps
            se = math.sqrt((1.0 / J + a["n_test"] / a["n_train"]) * var)
            return (mean - e) / se > _t_crit(alpha, J - 1)
        # scalar on either side: no paired variance available, so compare means +/- eps
        return sign * (_mean(x) - _mean(y)) > e

    def _dominates(self, a, b, cfg) -> bool:
        if not self._comparable(a, b, cfg):
            return False
        ms = cfg[0]
        return (any(self._better(a, b, m, cfg) for m in ms)
                and not any(self._better(b, a, m, cfg) for m in ms))

    def _equivalent(self, a, b, cfg) -> bool:
        if not self._comparable(a, b, cfg):
            return False
        return not any(self._better(a, b, m, cfg) or self._better(b, a, m, cfg) for m in cfg[0])
