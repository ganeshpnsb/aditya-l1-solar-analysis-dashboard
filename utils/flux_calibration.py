"""
flux_calibration.py
===================
Convert raw X-ray detector counts into approximate physical flux
(W m^-2 in the GOES 1-8 Angstrom soft X-ray band) and assign a GOES flare
class (A / B / C / M / X).

Why this exists
---------------
Detector "counts" are instrument-specific and not directly comparable across
payloads or with the wider heliophysics community. Operational solar-flare
nomenclature is expressed in the **GOES classification**, which is defined on
the peak 1-8 A soft X-ray irradiance:

    Class   Peak flux (W m^-2)
    -----   ------------------
    A       < 1e-7
    B       1e-7  ->  1e-6
    C       1e-6  ->  1e-5
    M       1e-5  ->  1e-4
    X       >= 1e-4

A real conversion requires the full instrument response (effective area,
energy redistribution matrix, live-time, distance scaling). We do not have that
here, so we expose a transparent, *configurable* linear-in-log calibration:

    flux = scale * (counts ** gamma)

`scale` and `gamma` are chosen so that a typical quiescent background maps to a
low A/B-class flux and bright flares reach M/X. The calibration is fully
documented and adjustable from the UI, so it is honest about being a proxy
rather than a black box.

Public API
----------
GOESCalibration                       : dataclass holding calibration constants
calibrate_counts(counts, calib)       : counts (array/scalar) -> flux (W m^-2)
flux_to_goes_class(flux)              : flux -> "M2.3" style GOES label
counts_to_goes_class(counts, calib)   : convenience counts -> GOES label
add_flux_columns(df, calib)           : return df copy with 'flux' + 'goes_class'
auto_calibration(background_counts)   : derive a calibration from a quiet level
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[float, int, np.ndarray, pd.Series]

# GOES class lower-bound thresholds in W m^-2 (1-8 Angstrom band).
GOES_THRESHOLDS = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}
_ORDERED_CLASSES = ["X", "M", "C", "B", "A"]  # high -> low for lookup


@dataclass
class GOESCalibration:
    """
    Parameters of the proxy counts -> flux conversion.

    flux = scale * (max(counts, 1) ** gamma)

    Attributes
    ----------
    scale : multiplicative constant (W m^-2).
    gamma : power-law exponent applied to counts.
    label : short human description of how it was derived.
    """

    scale: float = 1e-9
    gamma: float = 1.0
    label: str = "default"

    def as_dict(self) -> dict:
        return {"scale": self.scale, "gamma": self.gamma, "label": self.label}


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def calibrate_counts(counts: ArrayLike, calib: GOESCalibration | None = None) -> ArrayLike:
    """
    Convert counts to an approximate 1-8 A flux in W m^-2.

    Works on scalars, numpy arrays and pandas Series. Counts are floored at 1
    to keep the power-law well-defined.
    """
    calib = calib or GOESCalibration()
    arr = np.asarray(counts, dtype=float)
    arr = np.clip(arr, 1.0, None)
    flux = calib.scale * np.power(arr, calib.gamma)
    if np.isscalar(counts) or (hasattr(counts, "ndim") and arr.ndim == 0):
        return float(flux)
    if isinstance(counts, pd.Series):
        return pd.Series(flux, index=counts.index)
    return flux


def flux_to_goes_class(flux: float) -> str:
    """
    Map a flux value (W m^-2) to a GOES label such as ``C3.2`` or ``X1.0``.

    The number after the letter is the flux divided by the class' lower
    threshold (the standard GOES sub-class magnitude).
    """
    if flux is None or not np.isfinite(flux) or flux <= 0:
        return "A0.0"

    for letter in _ORDERED_CLASSES:
        threshold = GOES_THRESHOLDS[letter]
        if flux >= threshold:
            magnitude = flux / threshold
            # X-class is open-ended; others nominally span 1.0-9.9.
            return f"{letter}{magnitude:.1f}"
    # Below A-class threshold.
    magnitude = flux / GOES_THRESHOLDS["A"]
    return f"A{max(magnitude, 0.0):.1f}"


def goes_class_letter(flux: float) -> str:
    """Return just the class letter (A/B/C/M/X) for a flux value."""
    label = flux_to_goes_class(flux)
    return label[0] if label else "A"


def counts_to_goes_class(counts: float, calib: GOESCalibration | None = None) -> str:
    """Convenience: counts -> GOES label in one call."""
    return flux_to_goes_class(calibrate_counts(counts, calib))


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------
def add_flux_columns(df: pd.DataFrame, calib: GOESCalibration | None = None) -> pd.DataFrame:
    """
    Return a copy of ``df`` with two extra columns:
        flux       : calibrated 1-8 A flux (W m^-2)
        goes_class : per-sample GOES label
    """
    if df is None or df.empty or "counts" not in df:
        return df
    calib = calib or GOESCalibration()
    out = df.copy()
    out["flux"] = calibrate_counts(out["counts"], calib)
    out["goes_class"] = out["flux"].apply(flux_to_goes_class)
    return out


# ---------------------------------------------------------------------------
# Auto calibration
# ---------------------------------------------------------------------------
def auto_calibration(
    background_counts: float,
    background_class_flux: float = 5e-8,
    gamma: float = 1.0,
    label: str = "auto",
) -> GOESCalibration:
    """
    Derive a calibration so that the supplied quiescent ``background_counts``
    maps to ``background_class_flux`` (default ~mid A-class).

    Solving flux = scale * counts**gamma for scale:
        scale = background_class_flux / (background_counts ** gamma)
    """
    bg = max(float(background_counts), 1.0)
    scale = background_class_flux / (bg ** gamma)
    return GOESCalibration(scale=scale, gamma=gamma, label=label)
