"""
catalog_generator.py
====================
Combine the SoLEXS and HEL1OS flare catalogues into a single master catalogue.

Matching rule (per spec)
-------------------------
If a SoLEXS detection and a HEL1OS detection occur within +/- 60 seconds of each
other, they are treated as the *same* physical solar flare event. The merged row
records which instruments saw it, the combined peak counts, and a confidence
score that rewards multi-instrument agreement and close temporal coincidence.

Output columns
--------------
- Event Time
- SoLEXS Detection   (Yes / No)
- HEL1OS Detection   (Yes / No)
- Peak Counts        (max of the contributing peaks)
- Confidence Score   (0-1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MASTER_COLUMNS = [
    "Event Time",
    "SoLEXS Detection",
    "HEL1OS Detection",
    "Peak Counts",
    "SoLEXS Counts",
    "HEL1OS Counts",
    "Delta (s)",
    "Confidence Score",
]


def generate_master_catalog(
    solexs_cat: pd.DataFrame,
    hel1os_cat: pd.DataFrame,
    tolerance_s: float = 60.0,
) -> pd.DataFrame:
    """
    Cross-match two single-instrument catalogues into a master catalogue.

    Parameters
    ----------
    solexs_cat  : SoLEXS catalogue from ``flare_detector.detect_flares``.
    hel1os_cat  : HEL1OS catalogue.
    tolerance_s : coincidence window in seconds (default +/- 60 s).

    Returns
    -------
    Master catalogue DataFrame sorted by Event Time.
    """
    solexs = _prep(solexs_cat)
    hel1os = _prep(hel1os_cat)

    used_hel1os: set[int] = set()
    rows = []

    # ---- 1. Walk SoLEXS detections, attach the nearest HEL1OS match ---------
    for _, s_row in solexs.iterrows():
        match_idx, delta = _nearest(s_row["Peak Time"], hel1os, used_hel1os, tolerance_s)
        if match_idx is not None:
            h_row = hel1os.loc[match_idx]
            used_hel1os.add(match_idx)
            rows.append(
                _build_row(
                    event_time=_mean_time(s_row["Peak Time"], h_row["Peak Time"]),
                    solexs=True,
                    hel1os=True,
                    s_counts=s_row["Peak Counts"],
                    h_counts=h_row["Peak Counts"],
                    delta=delta,
                    tolerance_s=tolerance_s,
                )
            )
        else:
            rows.append(
                _build_row(
                    event_time=s_row["Peak Time"],
                    solexs=True,
                    hel1os=False,
                    s_counts=s_row["Peak Counts"],
                    h_counts=np.nan,
                    delta=np.nan,
                    tolerance_s=tolerance_s,
                )
            )

    # ---- 2. Any HEL1OS detections that were never matched -------------------
    for idx, h_row in hel1os.iterrows():
        if idx in used_hel1os:
            continue
        rows.append(
            _build_row(
                event_time=h_row["Peak Time"],
                solexs=False,
                hel1os=True,
                s_counts=np.nan,
                h_counts=h_row["Peak Counts"],
                delta=np.nan,
                tolerance_s=tolerance_s,
            )
        )

    master = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    if not master.empty:
        master = master.sort_values("Event Time").reset_index(drop=True)
    return master


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prep(cat: pd.DataFrame) -> pd.DataFrame:
    if cat is None or cat.empty:
        return pd.DataFrame(columns=["Peak Time", "Peak Counts"])
    out = cat.copy()
    out["Peak Time"] = pd.to_datetime(out["Peak Time"])
    return out.reset_index(drop=True)


def _nearest(target_time, candidates, used, tolerance_s):
    """Return (index, delta_seconds) of the closest unused candidate within tolerance."""
    if candidates.empty:
        return None, None
    best_idx, best_delta = None, None
    for idx, row in candidates.iterrows():
        if idx in used:
            continue
        delta = abs((row["Peak Time"] - target_time).total_seconds())
        if delta <= tolerance_s and (best_delta is None or delta < best_delta):
            best_idx, best_delta = idx, delta
    return best_idx, best_delta


def _mean_time(t1, t2):
    return t1 + (t2 - t1) / 2


def _build_row(event_time, solexs, hel1os, s_counts, h_counts, delta, tolerance_s):
    peak_counts = np.nanmax([v for v in [s_counts, h_counts] if not pd.isna(v)] or [np.nan])
    confidence = _confidence(solexs, hel1os, delta, tolerance_s, s_counts, h_counts)
    return {
        "Event Time": event_time,
        "SoLEXS Detection": "Yes" if solexs else "No",
        "HEL1OS Detection": "Yes" if hel1os else "No",
        "Peak Counts": round(float(peak_counts), 2) if not pd.isna(peak_counts) else np.nan,
        "SoLEXS Counts": round(float(s_counts), 2) if not pd.isna(s_counts) else np.nan,
        "HEL1OS Counts": round(float(h_counts), 2) if not pd.isna(h_counts) else np.nan,
        "Delta (s)": round(float(delta), 1) if not pd.isna(delta) else np.nan,
        "Confidence Score": confidence,
    }


def _confidence(solexs, hel1os, delta, tolerance_s, s_counts, h_counts):
    """
    Confidence score in [0, 1].

    * Dual-instrument coincidences score highest, scaled by how close in time the
      two detections were (closer => higher).
    * Single-instrument detections get a moderate base score scaled by amplitude.
    """
    if solexs and hel1os:
        temporal = 1.0 - (delta / tolerance_s) * 0.5 if not pd.isna(delta) else 0.8
        return round(float(np.clip(0.6 + 0.4 * temporal, 0.0, 1.0)), 3)

    # Single instrument: scale base confidence by signal strength.
    counts = s_counts if solexs else h_counts
    counts = 0.0 if pd.isna(counts) else counts
    amp_factor = float(np.clip(counts / 1000.0, 0.0, 1.0))
    return round(float(np.clip(0.35 + 0.25 * amp_factor, 0.0, 1.0)), 3)
