"""
Aditya-L1 Solar Flare Analysis Platform - utility package.

Modules
-------
fits_parser        : Read SoLEXS / HEL1OS FITS files into clean time-series.
flare_detector     : Smooth, threshold and peak-detect solar flares.
catalog_generator  : Cross-match instruments into a master catalogue.
forecasting        : Time-series flare-probability forecasting.
evaluation         : Skill metrics (TPR, FAR, lead time, F1, confusion matrix).
"""

__all__ = [
    "fits_parser",
    "flare_detector",
    "catalog_generator",
    "forecasting",
    "evaluation",
]
