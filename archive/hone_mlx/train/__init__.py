"""Training orchestration layer.

Public surface:

* :class:`Stage` — one entry of the full training sequence.
* :func:`full_sequence` — the canonical 5-stage sequence.
* :func:`select_stages` — resolve a ``--stages`` CLI selector.
* :class:`TrainOrchestrator` — run the sequence end-to-end with
  resume, marker rebuilds, and divergence detection.
* :class:`DivergenceDetector` — decide whether a stage log shows
  runaway NaN losses.
* :class:`StageMarker` — sidecar bookkeeping for prepared JSONLs.
* :class:`StageRunner` — wrap a launcher with per-stage tee capture.
"""

from hone.train.divergence import DivergenceDetector
from hone.train.marker import StageMarker
from hone.train.orchestrator import TrainOrchestrator
from hone.train.runner import StageRunner
from hone.train.stages import Stage, full_sequence, select_stages

__all__ = [
    "DivergenceDetector",
    "Stage",
    "StageMarker",
    "StageRunner",
    "TrainOrchestrator",
    "full_sequence",
    "select_stages",
]
