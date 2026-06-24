"""
flare_detector.py
=================
Detect solar flares in an X-ray light curve.

Algorithm (as specified)
-------------------------
1. Smooth the raw count series (rolling / Savitzky-Golay) to suppress noise.
2. Compute a moving-average baseline + rolling standard deviation.
3. Flag samples that rise above ``baseline + k * sigma`` (adaptive threshold).
4. Group consecutive above-threshold samples into events and pick the peak.

The result is a per-instrument flare catalogue with the columns requested by the
platform spec: Peak Index, Peak Counts, Source, Peak Time and Energy (when an
energy column / GOES-class estimate is available).

Public API
----------
detect_flares(df, ...)        -> catalogue DataFrame
build_catalogue(df, ...)      -> alias used by the UI
smooth_series(values, ...)    -> np.ndarray
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .flux_calibration import GOESCalibration, calibrate_counts, flux_to_goes_class

try:
    from scipy.signal import find_peaks, savgol_filter

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# Catalogue column order requested by the platform.
CATALOGUE_COLUMNS = [
    "Peak Index",
    "Peak Time",
    "Peak Counts",
    "Source",
    "Background",
    "Net Counts",
    "Peak Flux (W/m^2)",
    "Energy",
    "GOES Class",
]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def smooth_series(values: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    Smooth a 1-D signal.

    Uses a Savitzky-Golay filter when SciPy is available (preserves peak shape),
    otherwise falls back to a simple centred rolling mean.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return values

    window = max(5, min(window, values.size if values.size % 2 else values.size - 1))
    if window % 2 == 0:
        window += 1  # Savitzky-Golay requires an odd window

    if _HAS_SCIPY:
        poly = min(poly, window - 1)
        try:
            return savgol_filter(values, window_length=window, polyorder=poly)
        except Exception:
            pass

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_flares(
    df: pd.DataFrame,
    smooth_window: int = 11,
    baseline_window: int = 60,
    sigma: float = 3.0,
    min_separation_s: float = 60.0,
    min_prominence: Optional[float] = None,
    calib: Optional[GOESCalibration] = None,
) -> pd.DataFrame:
    """
    Detect flares in a single-instrument light curve.

    Parameters
    ----------
    df               : time-series with columns [time, seconds, counts, source].
    smooth_window    : smoothing window length (samples).
    baseline_window  : rolling window (samples) used for the moving-average baseline.
    sigma            : number of standard deviations above baseline for a detection.
    min_separation_s : minimum time gap (seconds) between two distinct peaks.
    min_prominence   : optional explicit peak prominence; auto-derived if None.

    Returns
    -------
    Catalogue DataFrame (see ``CATALOGUE_COLUMNS``).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=CATALOGUE_COLUMNS)

    df = df.sort_values("seconds").reset_index(drop=True)
    counts = df["counts"].to_numpy(dtype=float)
    seconds = df["seconds"].to_numpy(dtype=float)
    source = str(df["source"].iloc[0]) if "source" in df else "Unknown"

    # 1. Smooth ---------------------------------------------------------------
    smoothed = smooth_series(counts, window=smooth_window)

    # 2. Quiescent baseline + noise estimate ----------------------------------
    # A wide, low-quantile rolling window estimates the quiescent background
    # without being inflated by broad (soft X-ray) flares that a plain median
    # would otherwise absorb. Noise is estimated from the residual using a
    # robust MAD so a few bright flares don't dominate the threshold.
    s = pd.Series(smoothed)
    wide = max(baseline_window, smoothed.size // 8, 21)
    baseline = s.rolling(wide, center=True, min_periods=1).quantile(0.20)
    residual = smoothed - baseline.to_numpy()
    mad = np.median(np.abs(residual - np.median(residual))) or 1.0
    noise = 1.4826 * mad  # MAD -> sigma equivalent for a normal distribution
    threshold = (baseline + sigma * noise).to_numpy()

    # 3. Sample cadence (used to translate separation seconds -> samples) -----
    cadence = float(np.median(np.diff(seconds))) if seconds.size > 1 else 1.0
    cadence = cadence if cadence > 0 else 1.0
    distance = max(1, int(round(min_separation_s / cadence)))

    if min_prominence is None:
        min_prominence = max(2.0 * float(noise), 1e-6)

    # 4. Peak finding ---------------------------------------------------------
    peak_indices = _find_peaks(smoothed, threshold, distance, min_prominence)

    # GOES flux calibration: derive a calibration from the quiescent background
    # (median baseline) when the caller does not supply one explicitly.
    if calib is None:
        from .flux_calibration import auto_calibration

        bg_level = float(np.nanmedian(baseline.to_numpy()))
        calib = auto_calibration(bg_level, label=f"{source}-auto")

    rows = []
    for idx in peak_indices:
        bg = float(baseline.iloc[idx])
        peak_counts = float(counts[idx])
        net = max(peak_counts - bg, 0.0)
        # Calibrate the absolute peak counts to a physical 1-8 A flux, then
        # derive the GOES class from that flux (instead of a raw-count heuristic).
        peak_flux = float(calibrate_counts(peak_counts, calib))
        goes = flux_to_goes_class(peak_flux)
        energy = _estimate_energy(df, idx)
        rows.append(
            {
                "Peak Index": int(idx),
                "Peak Time": df["time"].iloc[idx],
                "Peak Counts": round(peak_counts, 2),
                "Source": source,
                "Background": round(bg, 2),
                "Net Counts": round(net, 2),
                "Peak Flux (W/m^2)": f"{peak_flux:.2e}",
                "Energy": energy,
                "GOES Class": goes,
            }
        )

    catalogue = pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)
    return catalogue


# Alias kept for readability in the UI layer.
build_catalogue = detect_flares


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _find_peaks(smoothed, threshold, distance, prominence):
    """Return indices of peaks that exceed the adaptive threshold."""
    if _HAS_SCIPY:
        peaks, _ = find_peaks(
            smoothed,
            height=threshold,
            distance=distance,
            prominence=prominence,
        )
        return peaks

    # ---- Pure-NumPy fallback peak detector ----
    above = smoothed > threshold
    peaks = []
    i = 1
    n = len(smoothed)
    last_peak = -distance
    while i < n - 1:
        if above[i] and smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            if i - last_peak >= distance:
                peaks.append(i)
                last_peak = i
        i += 1
    return np.asarray(peaks, dtype=int)


def _estimate_energy(df, idx):
    """
    Return an energy string for the peak.

    If the source DataFrame carries an explicit 'energy' column we report it in
    keV; otherwise we return 'N/A'. The GOES class is now derived from the
    calibrated soft X-ray flux (see flux_calibration), not from this function.
    """
    if "energy" in df.columns:
        try:
            return f"{float(df['energy'].iloc[idx]):.2f} keV"
        except Exception:
            pass
    return "N/A"
