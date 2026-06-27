"""
evaluation.py
=============
Skill / performance metrics for the flare forecasting system.

Given a set of predicted alerts and a set of observed (ground-truth) flares,
this module computes a confusion matrix and the standard verification scores:

    TPR (True Positive Rate / Recall) = TP / (TP + FN)
    FAR (False Alarm Rate)            = FP / (FP + TP)
    Accuracy                          = (TP + TN) / (TP + TN + FP + FN)
    Precision                         = TP / (TP + FP)
    Recall                            = TP / (TP + FN)
    F1                                = 2 * P * R / (P + R)
    Average Lead Time                 = mean(Flare Peak Time - Alert Time)

The matching between alerts and flares is done with a temporal tolerance window.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    """Container for all evaluation outputs."""

    tp: int
    fp: int
    fn: int
    tn: int
    tpr: float           # True Positive Rate / Recall
    far: float           # False Alarm Rate
    accuracy: float
    precision: float
    recall: float
    f1: float
    avg_lead_time_s: float
    lead_times_s: List[float]

    def as_dict(self) -> dict:
        return asdict(self)

    def confusion_matrix(self) -> pd.DataFrame:
        """Return the 2x2 confusion matrix as a labelled DataFrame."""
        return pd.DataFrame(
            [[self.tp, self.fn], [self.fp, self.tn]],
            index=["Actual: Flare", "Actual: No Flare"],
            columns=["Predicted: Flare", "Predicted: No Flare"],
        )

    def metric_cards(self) -> dict:
        """Friendly dict for rendering metric cards in the UI."""
        return {
            "True Positive Rate (TPR)": f"{self.tpr * 100:.1f}%",
            "False Alarm Rate (FAR)": f"{self.far * 100:.1f}%",
            "Average Lead Time": f"{self.avg_lead_time_s / 60:.1f} min",
            "Accuracy": f"{self.accuracy * 100:.1f}%",
            "Precision": f"{self.precision * 100:.1f}%",
            "Recall": f"{self.recall * 100:.1f}%",
            "F1 Score": f"{self.f1:.3f}",
        }


def evaluate_forecasts(
    predicted_alerts: pd.DataFrame,
    actual_flares: pd.DataFrame,
    tolerance_s: float = 1800.0,
    total_windows: Optional[int] = None,
) -> EvaluationResult:
    """
    Compare predicted alerts against observed flares.

    Parameters
    ----------
    predicted_alerts : DataFrame with an 'Alert Time' (or 'Peak Time') column.
    actual_flares    : DataFrame with a 'Peak Time' column.
    tolerance_s      : an alert counts as a hit if a real flare occurs within this
                       window after the alert (default 5 minutes).
    total_windows    : optional total number of evaluation windows, used to
                       estimate true negatives (TN). Defaults to a heuristic.

    Returns
    -------
    EvaluationResult
    """
    alerts = _times(predicted_alerts, ["Alert Time", "Peak Time", "time"])
    flares = _times(actual_flares, ["Peak Time", "time"])

    matched_flares: set[int] = set()
    lead_times: List[float] = []
    tp = 0
    fp = 0

    # ---- Match each alert to the nearest still-unmatched real flare ---------
    for a_time in alerts:
        best_j, best_lead = None, None
        for j, f_time in enumerate(flares):
            if j in matched_flares:
                continue
            lead = (f_time - a_time).total_seconds()
            # A valid alert precedes (or nearly coincides with) the flare.
            if -tolerance_s <= lead <= tolerance_s:
                if best_lead is None or abs(lead) < abs(best_lead):
                    best_j, best_lead = j, lead
        if best_j is not None:
            tp += 1
            matched_flares.add(best_j)
            lead_times.append(max(best_lead, 0.0))
        else:
            fp += 1

    fn = len(flares) - len(matched_flares)

    # ---- Estimate true negatives -------------------------------------------
    if total_windows is None:
        total_windows = (tp + fp + fn) * 3 + 10  # heuristic background of quiet windows
    tn = max(total_windows - tp - fp - fn, 0)

    # ---- Metrics ------------------------------------------------------------
    tpr = _safe_div(tp, tp + fn)
    far = _safe_div(fp, fp + tp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = tpr
    f1 = _safe_div(2 * precision * recall, precision + recall)
    avg_lead = float(np.mean(lead_times)) if lead_times else 0.0

    return EvaluationResult(
        tp=tp, fp=fp, fn=fn, tn=tn,
        tpr=round(tpr, 4), far=round(far, 4),
        accuracy=round(accuracy, 4), precision=round(precision, 4),
        recall=round(recall, 4), f1=round(f1, 4),
        avg_lead_time_s=round(avg_lead, 1), lead_times_s=lead_times,
    )


def evaluate_from_labels(y_true, y_pred) -> EvaluationResult:
    """
    Alternative evaluation when binary label arrays are available
    (e.g. from a train/test split of the forecaster).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    tpr = _safe_div(tp, tp + fn)
    far = _safe_div(fp, fp + tp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = tpr
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return EvaluationResult(
        tp=tp, fp=fp, fn=fn, tn=tn,
        tpr=round(tpr, 4), far=round(far, 4),
        accuracy=round(accuracy, 4), precision=round(precision, 4),
        recall=round(recall, 4), f1=round(f1, 4),
        avg_lead_time_s=0.0, lead_times_s=[],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


def _times(df, candidate_cols):
    if df is None or df.empty:
        return []
    for col in candidate_cols:
        if col in df.columns:
            return list(pd.to_datetime(df[col]).sort_values())
    return []
