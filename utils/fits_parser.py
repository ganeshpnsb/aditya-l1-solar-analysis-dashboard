"""
fits_parser.py
==============

ISROGENZ - Aditya-L1 Solar Flare Analysis Platform

This module is responsible for reading and processing FITS files obtained from
the Aditya-L1 mission's SoLEXS (Soft X-ray) and HEL1OS (Hard X-ray) payloads.

The parser extracts the required time-series information from uploaded FITS
files and converts it into a clean, analysis-ready pandas DataFrame. Since
different FITS products may contain variations in HDU structures and column
names, the parser searches through multiple candidate columns to identify the
time axis and count-rate values.

This implementation is designed exclusively for uploaded FITS files and does
not use synthetic or demo datasets. All analysis, visualization, flare
detection, and forecasting operations are performed using user-provided mission
data.

## Public API

parse_fits(file, source)
-> Returns a pandas DataFrame containing:
[time, seconds, counts, source]

load_timeseries(uploaded_file, source)
-> Loads and preprocesses uploaded FITS files for further analysis.

validate_fits(file)
-> Validates the uploaded FITS file and returns:
(is_valid: bool, message: str)

Developed by Team ISROGENZ
for the Bharatiya Antariksh Hackathon (BAH) 2026.
"""

from __future__ import annotations
import gzip
import io
import io
from datetime import datetime, timedelta
import streamlit as st
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except Exception:  # pragma: no cover - streamlit not installed or unavailable
    UploadedFile = object

# Astropy is optional at import time so the module degrades gracefully if the
# environment has not installed it yet. All real parsing requires it, though.
try:
    from astropy.io import fits
    from astropy.time import Time

    _HAS_ASTROPY = True
except Exception:  # pragma: no cover - environment without astropy
    _HAS_ASTROPY = False


# Candidate column names we will look for inside a FITS binary table HDU.
_TIME_COLUMNS = ["TIME", "time", "Time", "MJD", "mjd", "TSTART", "SECONDS"]
_COUNT_COLUMNS = [
    "COUNTS", "COUNT", "RATE", "COUNT_RATE",
    "counts", "rate", "FLUX", "flux", "PHA", "CPS", "cps",
    "ener", "ENER", "chn", "CHN"
]

# Recognised instrument labels.
SOURCE_SOLEXS = "SoLEXS"
SOURCE_HEL1OS = "HEL1OS"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_fits(file: Union[str, io.BytesIO, UploadedFile]) -> Tuple[bool, str]:
    """
    Lightweight validation performed *before* full parsing.

    Checks the file extension. Note: We intentionally allow files without the
    SIMPLE keyword through, since we parse with ignore_missing_simple=True.
    Returns a tuple of (is_valid, human_readable_message).
    """
    name = _get_name(file)
    if name and not name.lower().endswith((".fits", ".fit", ".fts", ".gz")):
        return False, f"'{name}' does not look like a FITS file (.fits/.fit/.fts)."

    try:
        file_bytes = _read_all_bytes(file)
        if not file_bytes:
            return False, "File is empty."
    except Exception as exc:  # pragma: no cover
        return False, f"Could not read file bytes: {exc}"

    # Allow files through regardless of SIMPLE keyword since we use ignore_missing_simple=True
    # during parsing. This is intentionally permissive to support non-standard FITS formats.
    return True, "File looks like a valid FITS candidate - will attempt full parsing."


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------
def parse_fits(
    file: Union[str, io.BytesIO, "UploadedFile"],
    source: str = SOURCE_SOLEXS,
) -> pd.DataFrame:
    """
    Parse a single FITS file into a tidy time-series DataFrame.

    Parameters
    ----------
    file   : path string, file-like object, or Streamlit UploadedFile.
    source : instrument label, either ``SoLEXS`` or ``HEL1OS``.

    Returns
    -------
    DataFrame with columns:
        time     : pandas datetime (absolute timestamp)
        seconds  : float seconds elapsed from the first sample
        counts   : float count rate / flux
        source   : instrument label
    """
    if not _HAS_ASTROPY:
        raise RuntimeError(
    "astropy is required to parse FITS files. Install it using `pip install astropy`."
)

    # Read file bytes using robust helper
    file_bytes = _read_all_bytes(file)

    # Handle gzip compression
    if getattr(file, "name", "").endswith(".gz"):
        try:
            file_bytes = gzip.decompress(file_bytes)
        except Exception as gz_exc:
            raise ValueError(f"Failed to decompress .gz file: {gz_exc}")
    
    buffer = io.BytesIO(file_bytes)

    # Try to open with lenient settings
    hdul = None
    try:
        hdul = fits.open(
            buffer,
            memmap=False,
            ignore_missing_simple=True,
            output_verify='silentfix+ignore',
        )
    except ValueError as ve:
        # If ignore_missing_simple doesn't work, try with even more lenient settings
        if "SIMPLE" in str(ve):
            buffer.seek(0)
            try:
                hdul = fits.open(
                    buffer,
                    memmap=False,
                    output_verify='silentfix+ignore',
                )
            except Exception:
                # If all else fails, try to manually fix the FITS structure
                buffer.seek(0)
                try:
                    # Read the file and attempt to add a minimal SIMPLE header if missing
                    file_bytes_mod = file_bytes
                    if not file_bytes_mod.startswith(b"SIMPLE"):
                        # Prepend a minimal SIMPLE header
                        simple_header = b"SIMPLE  =                    T / file does conform to FITS standard             "
                        file_bytes_mod = simple_header + file_bytes_mod
                    buffer = io.BytesIO(file_bytes_mod)
                    hdul = fits.open(buffer, memmap=False, output_verify='silentfix+ignore')
                except Exception as final_exc:
                    raise ValueError(f"Could not parse file with any method: {final_exc}") from ve
        else:
            raise
    
    with hdul:
        table_hdu = _find_table_hdu(hdul)
        if table_hdu is None:
            raise ValueError("No binary table HDU with usable columns was found.")

        colnames = list(table_hdu.columns.names)
        time_col = _first_match(colnames, _TIME_COLUMNS)
        count_col = _first_match(colnames, _COUNT_COLUMNS)

        if time_col is None or count_col is None:
            raise ValueError(
                f"Could not locate time/count columns. Available: {colnames}"
            )

        raw_time = np.asarray(table_hdu.data[time_col], dtype=float).ravel()
        counts = np.asarray(table_hdu.data[count_col], dtype=float).ravel()

        # Derive an absolute reference epoch from the header when possible.
        ref_epoch = _reference_epoch(table_hdu.header, hdul[0].header)
        times = _to_datetime(raw_time, ref_epoch, time_col)

    df = pd.DataFrame({"time": times, "counts": counts})
    df = df.dropna().sort_values("time").reset_index(drop=True)
    df["seconds"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["source"] = source
    return df[["time", "seconds", "counts", "source"]]


def load_timeseries(
    uploaded_file,
    source: str = SOURCE_SOLEXS,
) -> pd.DataFrame:
    return parse_fits(uploaded_file, source=source)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _find_table_hdu(hdul):
    """Return the first HDU that exposes a column collection."""
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None and hasattr(hdu, "columns"):
            try:
                if len(hdu.columns.names) >= 2:
                    return hdu
            except Exception:
                continue
    return None


def _first_match(available, candidates):
    """Return the first candidate present in ``available`` (case-insensitive)."""
    lower = {c.lower(): c for c in available}
    for cand in candidates:
        if cand in available:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _reference_epoch(*headers):
    """
    Try to read a reference epoch (MJDREF / DATE-OBS / TSTART) from any header
    so that relative second offsets can be converted to absolute timestamps.
    """
    for header in headers:
        if header is None:
            continue
        if "DATE-OBS" in header:
            try:
                return pd.to_datetime(header["DATE-OBS"])
            except Exception:
                pass
        if "MJDREF" in header and _HAS_ASTROPY:
            try:
                return pd.to_datetime(Time(float(header["MJDREF"]), format="mjd").datetime)
            except Exception:
                pass
    return None


def _to_datetime(raw_time, ref_epoch, time_col):
    """Convert a raw time array into absolute pandas timestamps."""
    # Heuristic: large values that look like MJD (~50000-70000) are treated as MJD.
    if time_col.upper().startswith("MJD") or (
        raw_time.size and 40000 < np.nanmedian(raw_time) < 80000
    ):
        if _HAS_ASTROPY:
            return pd.to_datetime(Time(raw_time, format="mjd").datetime)

    base = ref_epoch if ref_epoch is not None else pd.Timestamp("2024-01-01")
    return base + pd.to_timedelta(raw_time - np.nanmin(raw_time), unit="s")


def _as_buffer(file):
    """Normalise any supported input into something astropy.fits can open."""
    if isinstance(file, str):
        return file
    data = _read_all_bytes(file)
    return io.BytesIO(data)


def _read_all_bytes(file) -> bytes:
    if hasattr(file, "getvalue"):
        return file.getvalue()
    if hasattr(file, "read"):
        pos = file.tell() if hasattr(file, "tell") else None
        data = file.read()
        if pos is not None and hasattr(file, "seek"):
            file.seek(pos)
        return data
    raise TypeError("Unsupported file object passed to parser.")


def _peek_bytes(file, n: int) -> bytes:
    if isinstance(file, str):
        with open(file, "rb") as fh:
            return fh.read(n)
    data = _read_all_bytes(file)
    return data[:n]


def _get_name(file) -> Optional[str]:
    if isinstance(file, str):
        return file
    return getattr(file, "name", None)
