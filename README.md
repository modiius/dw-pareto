# pareto

A lightweight Pareto-front tracker for model evaluations, with a `click` CLI and a
self-contained HTML report.

One append-only JSONL ledger of runs. The front is recomputed from the ledger on
every call, so there is no stored front to drift and no incremental logic to get
wrong.

```
pixi install
pixi run demo                                   # build a throwaway ledger in ./demo
pixi run pareto -C demo report -o demo.html     # open it in a browser
pixi run test
```

## CLI

```bash
pareto -C experiments init -m f1:max -m latency_ms:min --eps f1=0.005 --alpha 0.05

pareto -C experiments log \
    -s f1=0.91,0.93,0.90 -s latency_ms=3.2 \
    --n-train 4000 --n-test 1000 \
    -c folds=8f3a2c -c hw=a100/bs32/8thr \
    --config-json params.json --artifact model.joblib --note "gbdt depth 7"
# 20260919-215548-79e6ad  on front

pareto -C experiments front                  # the front, as a table
pareto -C experiments runs --all             # every run with front / tied / dominated
pareto -C experiments show <run_id>          # scores, config, context, commit, who beat it
pareto -C experiments path <run_id>          # artifact directory
pareto -C experiments report -o front.html --open
```

`--root/-C` also reads `PARETO_ROOT`. Every command that computes a front accepts
`--metric`, `--eps`, `--alpha` and `--ignore` to ask a different question of the
same ledger without writing anything:

```bash
pareto -C experiments front -m f1:max                 # accuracy only
pareto -C experiments front --ignore hw               # across machines
pareto -C experiments front --eps f1=0.02             # only big wins count
```

A run's scores can be a scalar (`-s latency_ms=3.2`) or one value per CV fold
(`-s f1=0.91,0.93,0.90`, or `--scores-json` for anything larger).

## Library

```python
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
tr.explain()             # per run: status, who dominates it, what it ties with
tr.path(run_id)          # artifact dir; loading is your code's job
```

Restoring the code behind a row: `git worktree add ../restore <row["commit"]>`.

## How the front is decided

1. **Comparability.** Two runs are compared only if their `context` dicts are
   equal (minus any `--ignore` keys). Put the fold hash, the hardware, the dataset
   version - anything that makes numbers incommensurable - in `context`. Runs in
   different context groups both stay on the front; the CLI and the report say so.
2. **Better on a metric.** With per-fold lists on both sides, a one-sided
   Nadeau-Bengio corrected resampled t-test at `alpha`, testing the true
   difference against `eps` rather than against zero. Otherwise, means compared
   directly with the same `eps`.
3. **Domination.** `a` dominates `b` if it is better on at least one metric and
   `b` is better on none.
4. **Ties.** Among non-dominated runs, one that is indistinguishable from a run
   already kept is dropped. Ledger order is time order, so the earlier run wins -
   deliberately, so a re-run that lands on the same numbers does not churn the
   front.

`eps` is what keeps the front honest: without it, any noise is an "improvement"
and the front fills up. Set it to the smallest difference you would act on.

### What this is not

* **Not an inferential guarantee.** Every pair of runs is tested on every metric
  with no multiplicity correction, so the family-wise error rate is far above
  `alpha`. Front membership means "not yet ruled out", not "significantly best".
* **Ties are not transitive.** A~B and B~C does not give A~C, so step 4 above
  depends on ledger order. That is a deliberate tie-break, not a bug.
* **Mixed scalar/per-fold comparisons skip the test.** If one run logged a scalar
  and another logged folds for the same metric, they are compared by mean ± eps.
  Log folds on both sides if you want the test.

## Report

`pareto report` writes one HTML file with the data inlined - no server, no CDN,
nothing fetched at view time. It has a scatter of any two metrics with the front
envelope drawn as a staircase, a detail panel for the selected run (per-fold
spread, context, config, commit, artifact path, and which runs beat it), and a
sortable table of every run. Light and dark themes; the palette is validated for
colour-vision deficiency, and status is never carried by colour alone.

`--title` names the page, and `--embed` drops the `<html>/<head>/<body>` wrapper
so the same page can be pasted into a site or published as an artifact.

When three or more metrics define the front, some front runs necessarily sit
inside any 2-D envelope - they earn their place on a metric that is not on screen.
The report says so rather than drawing a misleading line through them.

## Layout

```
src/pareto/core.py       Tracker, fold_hash - the ledger and the statistics
src/pareto/cli.py        click commands
src/pareto/report.py     ledger -> HTML payload
src/pareto/template.html the report itself; the data is injected into one placeholder
scripts/demo.py          builds a throwaway ledger to look at
```

A full worked example - eleven scikit-learn classifiers scored over the same CV
folds, with the ledger committed alongside the code - lives in `../pareto-example`.
