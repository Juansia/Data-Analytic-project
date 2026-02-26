# Data-Analytic-project — Data-Driven Evaluation of Brent Crude Oil Forecasting Models

An **R-based time-series forecasting framework** for predicting **Brent crude oil closing prices** using **sentiment (SENT)** and **USDX** as exogenous inputs.  
Models are trained in **R** via **keras/TensorFlow**, and using **Grey Wolf Optimizer (GWO)** script computes **weighted ensemble** predictions in **R**.

---

## Overview

This repository supports:
- Multivariate forecasting with **BRENT Close** as the target (3-trading-day-ahead by default)
- Multiple deep-learning architectures (LSTM/GRU/BiLSTM/BiGRU, CNN hybrids, attention, encoder–decoder)
- Evaluation using **MAE, MSE, RMSE** (reported on a chronological held-out test split)
- Optional **GWO**-based ensemble weighting (non-negative weights that sum to 1)

---

## Project Structure

All R implementation files are located in:

```
Main/R Language/
├── dataset/                      # Input CSV lives here
│   └── processed_data_best_corr_sentiment.csv
├── train_models.R                # Train one model; saves metrics + predictions
├── gwo_ensemble.R                # (Optional) GWO weighted ensemble
└── outputs/                      # Generated outputs (auto-created)
```

---

## Installation

### Prerequisites
- R (recommended: 4.2+)
- RStudio
- TensorFlow backend (used through R `keras`)

### Install packages (first run only)

```r
install.packages(c("readr", "dplyr"))
install.packages("keras")
```

Install the TensorFlow backend (one-time):

```r
keras::install_keras()
```


## Data Requirements

`The dataset is obtained by following the procedures in the DATA COLLECTION folder`

Place the dataset CSV here:

`Main/R Language/dataset/processed_data_best_corr_sentiment.csv`

### Required columns
- `date` (YYYY-MM-DD)
- `BRENT Close` (target)
- `SENT` (sentiment feature)
- `USDX` 

Your prepared dataset in this repo typically contains additional columns (e.g., `BRENT Volume`, differences, etc.).  
The scripts will select the needed columns automatically based on the model/feature set.

Example (minimum) format:

```csv
date,BRENT Close,SENT,USDX
2023-01-03,82.45,0.65,101.23
2023-01-04,83.12,0.72,101.15
...
```

---

## Usage (RStudio)

1. Open the repository folder as an **RStudio Project**.
2. Set your working directory to:

   `Main/R Language/`

3. Train models (repeat for each model you want to evaluate).

### Train one model

Open `train_models.R` and set:

- `MODEL_NAME <- "SENT-Bi-GRU"` (example)

Then run:

```r
source("train_models.R")
```

Repeat for each `MODEL_NAME` you want to compare.

### (Optional) Run the GWO ensemble

After you have trained multiple models (so prediction CSVs exist in `outputs/`), run:

```r
source("gwo_ensemble.R")
```

---

## Available Models

Set `MODEL_NAME` in `train_models.R` to one of the following:

- `SENT-LSTM`
- `SENT-GRU`
- `SENT-Bi-LSTM`
- `SENT-Bi-GRU`
- `SENT-CNN-LSTM`
- `SENT-CNN-Bi-LSTM`
- `SENT-CNN-Bi-LSTM-Attention`
- `SENT-Encoder-decoder-LSTM`
- `SENT-USDX-Encoder-decoder-GRU`

---

## Outputs

All outputs are saved under:

`Main/R Language/outputs/`

### From `train_models.R`
- `best_scores_r.csv`  
  Appends one row per run with summary performance metrics (MAE, MSE, RMSE) and configuration fields.
- `pred_val_<MODEL>.csv` and `pred_test_<MODEL>.csv`  
  Per-sample predictions used by the ensemble script. Includes `target_date`, `y_true_orig`, and `y_pred_orig`.

### From `gwo_ensemble.R` (optional)
- `best_ensemble_weights_gwo.csv` — learned weights per model
- `ensemble_metrics_gwo.csv` — ensemble MAE/MSE/RMSE on the test set
- `ensemble_predictions_test_gwo.csv` — per-sample test predictions from the ensemble

---

## Notes for Reproducibility

- The scripts use **chronological splits** (no shuffling) and **fit scaling on training only** to reduce data leakage.
- If you change hyperparameters (epochs, lookback, search settings), document the settings you used when reporting results.

---

## Acknowledgments

- `keras` / TensorFlow for deep learning in R
- Grey Wolf Optimizer (GWO) metaheuristic approach for ensemble weight optimization
