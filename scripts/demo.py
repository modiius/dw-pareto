"""Build a throwaway ledger so `pareto report` has something to show.

    pixi run demo && pixi run pareto -C demo report -o demo.html --open
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from pareto import Tracker

ROOT = Path("demo")
FOLDS = "5x10-8f3a2c"          # stands in for fold_hash(...)
RNG = random.Random(0)

MODELS = [
    # name,            f1,    fold noise, latency, size
    ("logreg",         0.781, 0.020, 1.1, 0.4),
    ("logreg-l1",      0.784, 0.019, 1.1, 0.3),
    ("random-forest",  0.842, 0.017, 8.7, 41.0),
    ("hist-gbdt",      0.871, 0.015, 4.2, 12.0),
    ("hist-gbdt-deep", 0.874, 0.016, 9.6, 38.0),
    ("mlp-small",      0.859, 0.021, 2.4, 3.1),
    ("mlp-wide",       0.868, 0.022, 6.8, 22.0),
    ("svm-rbf",        0.848, 0.018, 31.0, 9.4),
    ("knn",            0.802, 0.024, 44.0, 96.0),
    ("stack",          0.883, 0.014, 18.5, 74.0),
]


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    tr = Tracker(ROOT, metrics={"f1": "max", "latency_ms": "min", "model_mb": "min"},
                 eps={"f1": 0.004, "latency_ms": 0.5, "model_mb": 1.0})
    for name, f1, noise, latency, size in MODELS:
        folds = [round(min(1.0, max(0.0, RNG.gauss(f1, noise))), 4) for _ in range(50)]
        res = tr.log(
            {"f1": folds, "latency_ms": latency, "model_mb": size},
            n_train=4000, n_test=1000,
            context={"folds": FOLDS, "hw": "a100/bs32/8thr"},
            config={"model": name, "seed": 0, "n_estimators": RNG.choice([100, 300, 500])},
            note=name,
            save=lambda d, n=name: (d / "model.txt").write_text(f"pretend weights for {n}\n"),
        )
        print(f"{name:16s} {'front' if res['on_front'] else '-':>6s}  {res['run_id']}")
    print(f"\n{len(tr.front())} of {len(tr.runs())} runs on the front -> pareto -C {ROOT} report --open")


if __name__ == "__main__":
    main()
