# Data-Analytic-project (R Version) — Data-Driven Evaluation of Brent Crude Oil Forecasting Models

An R-based time-series forecasting framework for Brent crude oil price prediction using **sentiment (SENT)** and optional **USDX** as exogenous inputs. The project trains multiple deep-learning architectures in **R** (via `keras`/TensorFlow) and supports a **Grey Wolf Optimizer (GWO)** weighted ensemble in **R**.

## 🎯 Overview

This project:
- Integrates **sentiment (SENT) and USDX** with Brent price history for multivariate forecasting
- Implements multiple deep learning architectures in **R** using `keras` (LSTM/GRU/BiLSTM/BiGRU/CNN hybrids, attention, encoder–decoder)
- Evaluates models using **MAE, MSE, RMSE**
- Optionally optimizes **ensemble weights** using an **R implementation of GWO**

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Requirements](#-data-requirements)
- [Usage (RStudio)](#-usage-rstudio)
- [Available Models](#-available-models)
- [Outputs](#-outputs)
- [Acknowledgments](#-acknowledgments)

## 📁 Project Structure

R implementation files are located in:

```
Main/R Language/
├── dataset/
│   └── processed_data_best_corr_sentiment.csv
├── train_models_best_only.R     # Train one model; append best scores to outputs/best_scores_r.csv
├── gwo_ensemble.R               # (Optional) GWO weighted ensemble in R
└── outputs/                     # Generated outputs (created automatically)
```

## 🚀 Installation

### Prerequisites
- R (recommended: R 4.2+)
- RStudio (recommended)
- TensorFlow backend (used through R `keras`)

### R package setup (first time)
Install required packages in R:

```r
install.packages(c("readr","dplyr"))
install.packages("keras")
```

Then install the TensorFlow backend (one-time):

```r
keras::install_keras()
```

## 📊 Data Requirements

Place your dataset CSV here:

`Main/R Language/dataset/processed_data_best_corr_sentiment.csv`

Required columns:
- `BRENT Close` (target)
- `SENT` (sentiment feature)
- `USDX` 

Example format:
```csv
Date,BRENT Close,SENT,USDX
2023-01-01,82.45,0.65,101.23
2023-01-02,83.12,0.72,101.15
...
```

## 💻 Usage (RStudio)

1) Open the repository folder as an **RStudio Project**.  
2) Set your working directory to: `Main/R Language/`  
3) Train models and record best scores.

### Single model training (best-score output only)
Open `train_models_best_only.R` and set:
- `MODEL_NAME <- "SENT-Bi-GRU"` (or another model)
- (optional) `LOOKBACK`, `EPOCHS`, `BATCH_SIZE`

Run in RStudio:

```r
source("train_models_best_only.R")
```

Repeat for each model you want to evaluate.

### (Optional) Weighted ensemble using GWO
After training multiple models, run:

```r
source("gwo_ensemble.R")
```

Edit `MODELS <- c(...)` inside `gwo_ensemble.R` to match the models you trained.

## 🤖 Available Models

Set `MODEL_NAME` in `train_models_best_only.R` to one of:

- `SENT-LSTM`
- `SENT-GRU`
- `SENT-Bi-LSTM`
- `SENT-Bi-GRU`
- `SENT-CNN-LSTM`
- `SENT-CNN-Bi-LSTM`
- `SENT-CNN-Bi-LSTM-Attention`
- `SENT-Encoder-decoder-LSTM`
- `SENT-USDX-Encoder-decoder-GRU`

## 📈 Outputs

All outputs are saved under:

`Main/R Language/outputs/`

### From `train_models_best_only.R`
- `best_scores_r.csv`  
  Appends one row per run with:
  - `MAE`, `MSE`, `RMSE` (original Brent price scale)
  - plus configuration fields like `model`, `features`, `lookback`, `horizon`

### From `gwo_ensemble.R` (optional)
- `best_ensemble_weights_gwo.csv`
- `ensemble_metrics_gwo.csv`
- `ensemble_predictions_test_gwo.csv`

## 🙏 Acknowledgments
- R `keras` / TensorFlow for deep learning in R
- Grey Wolf Optimizer (GWO) metaheuristic approach for ensemble weight optimization
