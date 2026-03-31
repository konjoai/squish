"""squish/squash/eval_binder.py — Bind lm_eval scores into the CycloneDX BOM.

Phase 2.  After running ``squish eval`` or the standalone lmeval script, call
:meth:`EvalBinder.bind` to populate
``components[0].modelCard.quantitativeAnalysis.performanceMetrics`` in the
sidecar written by Phase 1 (:mod:`squish.squash.sbom_builder`).

The update is *atomic*: written to a ``.tmp`` file then renamed so a partial
write never leaves a corrupt sidecar.  The operation is idempotent — calling
``bind`` twice with the same data produces the same number of entries
(overwrites, not appends).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class EvalBinder:
    """Mutate an existing CycloneDX ML-BOM sidecar with lm_eval task scores.

    All methods are static — the class is a namespace, not a stateful object.
    """

    @staticmethod
    def bind(
        bom_path: Path,
        lmeval_json_path: Path,
        baseline_path: Path | None = None,
    ) -> None:
        """Add or replace ``performanceMetrics`` entries in *bom_path*.

        Parameters
        ----------
        bom_path:
            Path to ``cyclonedx-mlbom.json`` written by Phase 1.
        lmeval_json_path:
            Path to a squish lmeval JSON result file, e.g.
            ``results/lmeval_Qwen2.5-1.5B-int4_20260323T034811.json``.
            Expected schema::

                {
                  "scores": {"arc_easy": 70.6, "arc_challenge": 43.6, ...},
                  "raw_results": {
                    "arc_easy": {"acc_norm_stderr,none": 0.0204, ...},
                    ...
                  }
                }

        baseline_path:
            Optional path to a second squish lmeval JSON representing the
            higher-precision reference (e.g. BF16).  When provided every
            metric entry gains a ``deltaFromBaseline`` key formatted as
            ``"+3.4"`` or ``"-1.2"``.

        Raises
        ------
        FileNotFoundError
            If *bom_path* or *lmeval_json_path* does not exist.
        json.JSONDecodeError
            If any of the JSON files is malformed.
        """
        bom: dict = json.loads(bom_path.read_text())
        lmeval: dict = json.loads(lmeval_json_path.read_text())

        baseline_scores: dict[str, float] | None = None
        if baseline_path is not None:
            baseline_scores = json.loads(baseline_path.read_text()).get("scores", {})

        scores: dict[str, float] = lmeval.get("scores", {})
        raw_results: dict = lmeval.get("raw_results", {})

        metrics: list[dict] = []
        for task, score in scores.items():
            entry: dict = {
                "type": "accuracy",
                "value": str(round(score, 1)),
                "slice": task,
            }

            # Confidence interval from acc_norm_stderr,none (0–1 fraction).
            # Omitted silently when the key is absent — never crash.
            raw_task: dict = raw_results.get(task, {})
            stderr_frac: float | None = raw_task.get("acc_norm_stderr,none")
            if stderr_frac is not None:
                half = round(1.96 * stderr_frac * 100, 1)
                entry["confidenceInterval"] = {
                    "lowerBound": str(round(score - half, 1)),
                    "upperBound": str(round(score + half, 1)),
                }

            if baseline_scores is not None and task in baseline_scores:
                delta = round(score - baseline_scores[task], 1)
                sign = "+" if delta >= 0 else ""
                entry["deltaFromBaseline"] = f"{sign}{delta}"

            metrics.append(entry)

        # Overwrite any prior metrics — idempotent, not accumulative.
        component: dict = bom["components"][0]
        component["modelCard"]["quantitativeAnalysis"]["performanceMetrics"] = metrics

        # Atomic write: .tmp → rename into place.
        tmp = bom_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(bom, indent=2))
            tmp.rename(bom_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        log.debug(
            "EvalBinder: wrote %d performanceMetrics entries to %s",
            len(metrics),
            bom_path,
        )
