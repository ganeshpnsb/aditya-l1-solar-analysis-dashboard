"""
forecasting.py
=============
Time-series forecasting of solar-flare probability.

Goal
----
Given a light curve, estimate the probability that a flare will occur within the
next ``N`` minutes, and translate that probability into a Low / Medium / High
risk level.

Approach
--------
We engineer rolling statistical features from the count series (level, slope,
rolling mean/std, recent maximum) and train a gradient-boosted classifier. The
label for each timestamp is 1 if a detected flare peak falls inside the
[t, t + horizon] window.

Model preference order:
    1. XGBoost (XGBClassifier)            - if installed
    2. scikit-learn GradientBoostingClassifier - fallback
    3. Logistic heuristic                 - if neither model can be trained

An optional TensorFlow LSTM hook is provided but disabled by default.

Public API
----------
FlareForecaster                      : trainable model class
build_features(df, horizon_min)      : feature/label engineering
forecast_probability(...)            : convenience one-shot prediction
risk_level(prob)                     : map probability -> risk band
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def _tensorflow_available() -> bool:
    """Lazy check for TensorFlow so importing this module never requires it."""
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:  # pragma: no cover - TF is heavy / optional
        return False


_HAS_TF = _tensorflow_available()


FEATURE_COLUMNS = [
    "counts",
    "roll_mean",
    "roll_std",
    "slope",
    "recent_max",
    "delta_from_mean",
]

RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    horizon_min: int = 30,
    flare_catalogue: Optional[pd.DataFrame] = None,
    window: int = 30,
) -> pd.DataFrame:
    """
    Build a supervised feature matrix from a light curve.

    Each row gets rolling features plus a binary label indicating whether a
    flare peak occurs within ``horizon_min`` minutes after that sample.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["label"])

    df = df.sort_values("seconds").reset_index(drop=True)
    counts = df["counts"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["counts"] = counts
    feat["roll_mean"] = counts.rolling(window, min_periods=1).mean()
    feat["roll_std"] = counts.rolling(window, min_periods=1).std().fillna(0.0)
    feat["slope"] = counts.diff().fillna(0.0)
    feat["recent_max"] = counts.rolling(window, min_periods=1).max()
    feat["delta_from_mean"] = counts - feat["roll_mean"]

    feat["label"] = _make_labels(df, horizon_min, flare_catalogue)
    feat["seconds"] = df["seconds"].values
    feat["time"] = df["time"].values
    return feat


def _make_labels(df, horizon_min, flare_catalogue):
    """1 if a flare peak occurs within the horizon ahead of each sample."""
    horizon_s = horizon_min * 60
    seconds = df["seconds"].to_numpy()
    labels = np.zeros(len(df), dtype=int)

    if flare_catalogue is not None and not flare_catalogue.empty:
        # Convert catalogue peak times to elapsed seconds against this curve.
        t0 = df["time"].iloc[0]
        peak_secs = (
            pd.to_datetime(flare_catalogue["Peak Time"]) - t0
        ).dt.total_seconds().to_numpy()
    else:
        # Self-label: treat local maxima well above the mean as flare peaks.
        peak_secs = _auto_peak_seconds(df)

    for ps in peak_secs:
        mask = (seconds <= ps) & (seconds >= ps - horizon_s)
        labels[mask] = 1
    return labels


def _auto_peak_seconds(df):
    counts = df["counts"].to_numpy()
    thr = counts.mean() + 2.5 * counts.std()
    idx = np.where(counts > thr)[0]
    return df["seconds"].to_numpy()[idx]


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------
@dataclass
class FlareForecaster:
    """A trainable flare-probability forecaster with graceful fallbacks."""

    horizon_min: int = 30
    prefer_lstm: bool = False          # opt-in TensorFlow LSTM backend
    seq_len: int = 24                  # look-back window (samples) for the LSTM
    model: object = field(default=None, init=False)
    scaler: object = field(default=None, init=False)
    backend: str = field(default="heuristic", init=False)
    trained: bool = field(default=False, init=False)

    def train(self, df: pd.DataFrame, flare_catalogue: Optional[pd.DataFrame] = None):
        """Fit the model on engineered features from ``df``."""
        feat = build_features(df, self.horizon_min, flare_catalogue)
        if feat.empty:
            return self

        X = feat[FEATURE_COLUMNS].to_numpy()
        y = feat["label"].to_numpy()

        # Need both classes present to train a classifier.
        if len(np.unique(y)) < 2 or not (_HAS_XGB or _HAS_SK or (self.prefer_lstm and _HAS_TF)):
            self.backend = "heuristic"
            self.trained = True
            self._heuristic_ref = (float(feat["counts"].mean()), float(feat["counts"].std() or 1.0))
            return self

        # Optional TensorFlow LSTM path (opt-in). Falls through to the
        # tree-based models on any failure so the app never hard-fails.
        if self.prefer_lstm and _HAS_TF:
            try:
                self._train_lstm(X, y)
                self.trained = True
                return self
            except Exception:  # pragma: no cover - degrade gracefully
                self.backend = "heuristic"

        if _HAS_SK:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)

        if _HAS_XGB:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                eval_metric="logloss",
                n_jobs=2,
            )
            self.backend = "xgboost"
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.08
            )
            self.backend = "sklearn"

        self.model.fit(X, y)
        self.trained = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return the per-sample probability of a flare within the horizon."""
        feat = build_features(df, self.horizon_min)
        if feat.empty:
            return np.array([])
        X = feat[FEATURE_COLUMNS].to_numpy()

        if self.backend == "lstm" and self.model is not None:
            return self._predict_lstm(X)

        if self.backend in ("xgboost", "sklearn") and self.model is not None:
            if self.scaler is not None:
                X = self.scaler.transform(X)
            return self.model.predict_proba(X)[:, 1]

        # Heuristic: logistic response on the standardised count level.
        mean, std = getattr(self, "_heuristic_ref", (feat["counts"].mean(), feat["counts"].std() or 1.0))
        z = (feat["counts"].to_numpy() - mean) / (std if std else 1.0)
        return 1.0 / (1.0 + np.exp(-(z - 1.5)))

    def forecast_next(self, df: pd.DataFrame) -> float:
        """Single scalar probability for the most recent sample(s)."""
        probs = self.predict_proba(df)
        if probs.size == 0:
            return 0.0
        # Use the trailing window average to reduce single-sample jitter.
        tail = probs[-max(1, min(5, probs.size)):]
        return float(np.clip(tail.mean(), 0.0, 1.0))

    # ---- TensorFlow LSTM backend -------------------------------------------
    def _build_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Turn a (n_samples, n_features) matrix into overlapping look-back
        sequences of shape (n_windows, seq_len, n_features). When ``y`` is
        provided the label of the *last* timestep in each window is returned.
        """
        seq_len = max(2, int(self.seq_len))
        if X.shape[0] <= seq_len:
            seq_len = max(2, X.shape[0] - 1)
        windows, labels = [], []
        for end in range(seq_len, X.shape[0] + 1):
            windows.append(X[end - seq_len:end])
            if y is not None:
                labels.append(y[end - 1])
        seqs = np.asarray(windows, dtype="float32")
        if y is not None:
            return seqs, np.asarray(labels, dtype="float32"), seq_len
        return seqs, seq_len

    def _train_lstm(self, X: np.ndarray, y: np.ndarray):
        """Train a small LSTM classifier on look-back sequences."""
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        seqs, labels, used_len = self._build_sequences(Xs, y)
        self.seq_len = used_len
        if seqs.shape[0] < 8 or len(np.unique(labels)) < 2:
            raise ValueError("Not enough sequence data to train an LSTM.")

        n_features = seqs.shape[2]
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(used_len, n_features)),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        # Weight the positive (flare) class up to counter class imbalance.
        pos = float(labels.sum())
        neg = float(len(labels) - pos)
        class_weight = {0: 1.0, 1: max(neg / max(pos, 1.0), 1.0)}

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            seqs, labels,
            epochs=12, batch_size=32, verbose=0,
            class_weight=class_weight,
        )
        self.model = model
        self.backend = "lstm"

    def _predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Predict per-sample probabilities, padding the early warm-up region."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        seqs, used_len = self._build_sequences(X)
        if seqs.shape[0] == 0:
            return np.zeros(X.shape[0])
        preds = self.model.predict(seqs, verbose=0).ravel()
        # The first (used_len - 1) samples have no full look-back window; pad
        # them with the first available prediction so output length matches.
        pad = np.full(used_len - 1, preds[0])
        return np.clip(np.concatenate([pad, preds]), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def forecast_probability(
    df: pd.DataFrame,
    horizon_min: int = 30,
    flare_catalogue: Optional[pd.DataFrame] = None,
    prefer_lstm: bool = False,
):
    """
    One-shot training + prediction. Returns (probability, risk_level, forecaster).

    Set ``prefer_lstm=True`` to use the optional TensorFlow LSTM backend (falls
    back automatically to gradient boosting / heuristic if TF is unavailable).
    """
    forecaster = FlareForecaster(horizon_min=horizon_min, prefer_lstm=prefer_lstm)
    forecaster.train(df, flare_catalogue)
    prob = forecaster.forecast_next(df)
    return prob, risk_level(prob), forecaster


def risk_level(prob: float, medium: float = 0.4, high: float = 0.7) -> str:
    """Map a probability into a Low / Medium / High risk band."""
    if prob >= high:
        return RISK_HIGH
    if prob >= medium:
        return RISK_MEDIUM
    return RISK_LOW


def probability_timeline(
    df: pd.DataFrame,
    horizon_min: int = 30,
    flare_catalogue: Optional[pd.DataFrame] = None,
    prefer_lstm: bool = False,
    forecaster: Optional["FlareForecaster"] = None,
) -> pd.DataFrame:
    """
    Return a per-timestamp probability series for plotting.

    Pass an already-trained ``forecaster`` to avoid retraining (e.g. to reuse
    the model produced by ``forecast_probability``).
    """
    if forecaster is None:
        forecaster = FlareForecaster(horizon_min=horizon_min, prefer_lstm=prefer_lstm)
        forecaster.train(df, flare_catalogue)
    probs = forecaster.predict_proba(df)
    out = df[["time", "seconds"]].copy().iloc[: len(probs)]
    out["probability"] = probs
    return out
