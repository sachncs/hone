"""Hyperparameter search orchestration.

Public surface:

* :class:`TrialSpec` — one hyperparameter combination.
* :class:`TrialResult` — one trial's outcome.
* :func:`expand_search` — Cartesian product over the search space.
* :func:`loss_from_output` — extract the lowest validation loss.
* :class:`TrialRunner` — train + benchmark one trial in isolation.
* :func:`select_best` — pick the trial that minimises / maximises
  the chosen objective.
"""

from hone.tune.runner import TrialRunner
from hone.tune.search import TrialSpec, expand_search, select_best
from hone.tune.spec import TrialResult, loss_from_output

__all__ = [
    "TrialResult",
    "TrialRunner",
    "TrialSpec",
    "expand_search",
    "loss_from_output",
    "select_best",
]
