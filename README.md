# Aditya-L1 Solar Flare Analysis Platform

A dark, ISRO / deep-space themed **Streamlit** dashboard that analyses
**SoLEXS** (soft X-ray) and **HEL1OS** (hard X-ray) FITS files from ISRO's
**Aditya-L1** solar observatory. It detects solar flares, builds a master
catalogue, forecasts flare probability, and reports verification metrics.

## Features

1. **Home Dashboard** - mission overview + metric cards (files, detected flares, predicted flares, last analysis).
2. **FITS Upload** - validated upload of SoLEXS / HEL1OS FITS files (name, size, status).
3. **Data Visualization** - soft / hard / combined light curves with zoom, hover tooltips and download-as-PNG.
4. **Solar Flare Detection** - smoothing -> moving-average baseline -> adaptive peak detection, with per-instrument catalogues and CSV download.
5. **Master Catalogue** - cross-matches detections within +/- 60 s into single events with a confidence score (`master_catalog.csv`).
6. **Forecasting** - trains an XGBoost / scikit-learn classifier to predict flare probability over the next N minutes, shown as a gauge + probability bar + timeline.
7. **Evaluation** - TPR, FAR, average lead time, accuracy, precision, recall, F1 and a confusion matrix.
8. **Alert System** - triggers a Solar Flare Alert when probability exceeds a configurable threshold.
9. **Download Center** - export every catalogue and report.

## Tech stack

- **Frontend:** Streamlit + Plotly
- **Data / science:** NumPy, Pandas, SciPy, Astropy
- **Machine learning:** scikit-learn, XGBoost 

## Getting started

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

Then open http://localhost:8501.



