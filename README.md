# Data-Analytic-project
Brent Forecasting Model
# Data-Driven Evaluation of Brent Crude Oil Forecasting Models

An advanced machine learning framework for Brent crude oil price prediction that integrates market sentiment analysis with deep learning architectures, featuring hyperparameter optimization via Grey Wolf Optimizer (GWO) and weighted ensemble methods.

## 🎯 Overview

This project implements a state-of-the-art time series forecasting system that:
- Integrates **sentiment analysis** from market news/reports with price data
- Utilizes 9 different deep learning architectures optimized for financial time series
- Employs Grey Wolf Optimizer (GWO) for automatic hyperparameter tuning
- Supports both single model and weighted ensemble predictions
- Provides **future price predictions** beyond the dataset
- Handles multivariate time series with external features (BRENT prices, USDX, Sentiment scores)

## 📋 Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Web App (Flask)](#-web-app-flask)
- [Available Models](#available-models)
- [Data Requirements](#data-requirements)
- [Configuration](#configuration)
- [Future Predictions](#future-predictions)
- [Results](#results)
- [HPC Execution](#hpc-execution)
- [Contributing](#contributing)
- [License](#license)

## ✨ Key Features

- **Sentiment Analysis Integration**: Incorporates market sentiment scores to enhance prediction accuracy
- **Future Price Forecasting**: Predict Brent prices up to 30+ days into the future
- **Multiple Deep Learning Models**: 9 architectures including LSTM, BiLSTM, GRU, CNN-LSTM with attention mechanisms
- **Dual Framework Support**: Implementations in both PyTorch and Keras/TensorFlow
- **Automated Hyperparameter Optimization**: Grey Wolf Optimizer for intelligent parameter search
- **Ensemble Learning**: Weighted ensemble with validation-based weight optimization
- **Feature Engineering**: Automatic feature selection from sentiment, USDX, and price data
- **Comprehensive Evaluation**: Multiple metrics (MAE, MSE, RMSE) with visualizations
- **Production Ready**: Model checkpointing and deployment-ready predictions

## 📁 Project Structure

```
Brent/
├── _keras/                     # Keras/TensorFlow implementations
│   ├── models/
│   │   ├── CNN_BiLSTM_ATTENTION.py  # CNN-BiLSTM with attention mechanism
│   │   └── keras_lstm.py            # Various LSTM architectures
│   └── trainer.py
├── pytorch/                    # PyTorch implementations
│   ├── models/
│   │   ├── lstm.py            # LSTM variants (standard, bidirectional, CNN)
│   │   └── gru.py             # GRU variants
│   └── trainer.py
├── config/                     # Configuration files
│   └── args.py                # Argument definitions and feature sets
├── data/                       # Data processing
│   └── data.py                # Data loading, preprocessing, normalization
├── fitness/                    # Fitness functions
│   └── fitness.py             # Solution decoding and result saving
├── utils/                      # Utility functions
│   ├── evaluation.py          # Metrics calculation
│   ├── helper.py              # Helper functions including future prediction
│   └── pytorchtools.py        # PyTorch utilities (early stopping)
├── scripts/                    # SLURM job scripts for HPC
│   ├── Bi-LSTM.sh
│   ├── Bi_GRU.sh
│   └── ...
├── main.py                     # Main entry point
├── run_ensemble.py            # Ensemble model execution
└── weighted_ensemble_model.py  # Weighted ensemble implementation
```

### Web app files

If you are running the Flask web UI, these files/folders are also expected at the project root:

```
app.py                 # Flask web app (UI + /predict endpoint)
prediction_module.py   # Core forecasting logic used by app.py
templates/
└── index.html         # Web UI (rendered at /)
```


## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)

### Setup


Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Single Model Training

To train a single model with hyperparameter optimization:

```bash
python main.py [MODEL_NAME]
```

Example:
```bash
python main.py Bi-GRU
python main.py CNN-BiLSTM-att
python main.py torch-CNN-LSTM
```

### Ensemble Model

To run the weighted ensemble model:

```bash
python main.py --run_ensemble
```

Or:
```bash
python main.py ensemble
```

### Quick Example

```bash
# Train a model
python main.py Bi-LSTM

# Run ensemble
python main.py ensemble
```


### 🌐 Web App (Flask)

This repo includes a simple Flask web interface that renders a homepage (`/`) and provides a prediction endpoint (`/predict`).

#### Requirements
- Python 3.8+
- `prediction_module.py` present in the project root
- A template file at `templates/index.html`

Install Flask (or install everything via `requirements.txt` if you have one):

```bash
pip install flask
# or:
pip install -r requirements.txt
```

#### Run the web app

```bash
python app.py
```

Open in your browser:
- http://127.0.0.1:5000/

#### API: `POST /predict`

Form fields:
- `days`: forecast horizon (supported: **3**, **5**, **22**; other values will be adjusted)
- `commodity`: **brent** or **sugar** (defaults to `brent`)

Example:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d "days=5" \
  -d "commodity=brent"
```


## 🤖 Available Models

### PyTorch Models
- **LSTM**: Standard Long Short-Term Memory
- **Bi-LSTM**: Bidirectional LSTM with dual FC layers
- **GRU**: Gated Recurrent Unit (bidirectional)
- **Bi-GRU**: Bidirectional GRU optimized for time series
- **torch-CNN-LSTM**: CNN + Bidirectional LSTM with attention mechanism

### Keras/TensorFlow Models
- **keras-cnn-bilstm**: CNN + Bidirectional LSTM with batch normalization
- **CNN-BiLSTM-att**: CNN + BiLSTM + Custom Attention Layer
- **encoder-decoder-LSTM**: Sequence-to-sequence LSTM architecture
- **encoder-decoder-GRU**: Sequence-to-sequence GRU architecture

## 📊 Data Requirements

### Dataset Format

The system expects data in CSV format (`dataset/processed_data_best_corr_sentiment.csv`) with:
- Date index
- **BRENT Close**: Brent crude oil closing prices
- **SENT**: Sentiment scores from market analysis
- **USDX**: US Dollar Index (optional)
- Additional technical indicators (optional)

Example data structure:
```csv
Date,BRENT Close,SENT,USDX
2023-01-01,82.45,0.65,101.23
2023-01-02,83.12,0.72,101.15
...
```

### Feature Sets

The system supports various feature combinations:

```python
Features = {
    0: 'BRENT Close',                    # Price only
    10: ['SENT', 'BRENT Close'],         # Sentiment + Price
    30: ['USDX', 'BRENT Close'],         # USDX + Price
    130: ['SENT', 'USDX', 'BRENT Close'] # All features
}
```

## ⚙️ Configuration

### Optimization Parameters

GWO optimization parameters:
- Population size: 10
- Iterations: 20
- Search space: 7 dimensions

### Hyperparameter Search Space

- **Optimizer**: ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'Adam', 'Adamax']
- **Learning rate**: [0.0001, 0.01]
- **Hidden units**: [2, 128] (powers of 2)
- **Dropout**: [0.2, 0.5]
- **Sequence length**: [3, 30]
- **Weight decay**: [0, 0.1]

### Model Configuration

Modify default settings in `config/args.py`:

```python
args.pred_len = 3          # Prediction horizon
args.seq_len = 20          # Input sequence length
args.epoch = 2000          # Training epochs
args.batch_size = 32       # Batch size
args.patience = 30         # Early stopping patience
args.features = ['BRENT Close', 'SENT']  # Default features
```


## 📈 Results

Results are automatically saved in the `./results/` directory:

- `best_scores.xlsx`: Best hyperparameters and scores for each model
- `[MODEL_NAME].csv`: Detailed training history
- `ensemble_results.xlsx`: Ensemble model performance
- `future_predictions/`: Future price forecasts
- Model checkpoints in `./checkpoints/`

### Performance Metrics

The system evaluates models using:
- **MSE** (Mean Squared Error) - Primary optimization metric
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

### Sample Results Output

```
Bi-GRU Model Results:
- Best MSE: 0.0156
- Best MAE: 0.0892
- Best RMSE: 0.1249
- Optimal sequence length: 22
- Optimal hidden units: 64
- Selected features: ['SENT', 'BRENT Close']
```


## 🙏 Acknowledgments

- [Mealpy](https://github.com/thieu1995/mealpy) for metaheuristic optimization
- Financial data providers
- Sentiment analysis data sources
