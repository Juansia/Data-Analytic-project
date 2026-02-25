# train_models_best_only.R
# R-only implementation (keras/TensorFlow) for Brent 3-day-ahead forecasting with SENT (and optional USDX).
# OUTPUT MODE: BEST SCORE ONLY
# - Saves ONLY one row of summary metrics per run (no predictions, no .h5 model).
# - Appends results into outputs/best_scores_r.csv
#
# Supported MODEL_NAME values:
#  - SENT-LSTM
#  - SENT-GRU
#  - SENT-Bi-LSTM
#  - SENT-Bi-GRU
#  - SENT-CNN-LSTM                 (keras substitute for "Torch CNN-LSTM")
#  - SENT-CNN-Bi-LSTM
#  - SENT-CNN-Bi-LSTM-Attention
#  - SENT-Encoder-decoder-LSTM
#  - SENT-USDX-Encoder-decoder-GRU

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(keras)
  library(tensorflow)
})

# ----------------------------
# CONFIG (edit these)
# ----------------------------
DATA_PATH  <- "dataset/processed_data_best_corr_sentiment.csv"
MODEL_NAME <- "SENT-Bi-GRU"
HORIZON    <- 3
LOOKBACK   <- 30
EPOCHS     <- 30
BATCH_SIZE <- 32
SEED       <- 42

# Where to save results (only best scores)
OUT_DIR <- "outputs"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
BEST_SCORES_FILE <- file.path(OUT_DIR, "best_scores_r.csv")

set.seed(SEED)
tensorflow::tf$random$set_seed(SEED)

# ----------------------------
# Feature sets by model name
# ----------------------------
get_features_for_model <- function(model_name) {
  if (model_name == "SENT-USDX-Encoder-decoder-GRU") {
    return(c("BRENT Close", "SENT", "USDX"))
  }
  return(c("BRENT Close", "SENT"))
}

# ----------------------------
# Helpers
# ----------------------------
minmax_fit <- function(df, cols) {
  mn <- sapply(df[cols], min, na.rm = TRUE)
  mx <- sapply(df[cols], max, na.rm = TRUE)
  list(min = mn, max = mx)
}

minmax_transform <- function(df, cols, scaler) {
  out <- df
  for (c in cols) {
    out[[c]] <- (out[[c]] - scaler$min[[c]]) / (scaler$max[[c]] - scaler$min[[c]] + 1e-9)
  }
  out
}

minmax_inverse_one <- function(x, col, scaler) {
  x * (scaler$max[[col]] - scaler$min[[col]] + 1e-9) + scaler$min[[col]]
}

split_chrono <- function(df, train_frac = 0.8, val_frac = 0.1) {
  n <- nrow(df)
  i_train <- floor(train_frac * n)
  i_val   <- floor((train_frac + val_frac) * n)
  list(
    train = df[1:i_train, ],
    val   = df[(i_train + 1):i_val, ],
    test  = df[(i_val + 1):n, ]
  )
}

make_sequences <- function(df, feature_cols, target_col, lookback, horizon) {
  X_list <- list()
  y_list <- c()

  n <- nrow(df)
  last_start <- n - lookback - horizon + 1
  if (last_start < 1) stop("Not enough rows for lookback + horizon. Reduce LOOKBACK or add more data.")

  for (i in 1:last_start) {
    x_win <- as.matrix(df[i:(i + lookback - 1), feature_cols, drop = FALSE])
    y_val <- df[[target_col]][i + lookback + horizon - 1]
    X_list[[length(X_list) + 1]] <- x_win
    y_list <- c(y_list, y_val)
  }

  X <- array(unlist(X_list), dim = c(length(X_list), lookback, length(feature_cols)))
  y <- array(y_list, dim = c(length(y_list), 1))
  list(X = X, y = y)
}

metrics <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  mae  <- mean(abs(y_pred - y_true))
  mse  <- mean((y_pred - y_true)^2)
  rmse <- sqrt(mse)
  c(MAE = mae, MSE = mse, RMSE = rmse)
}

# ----------------------------
# Model builders
# ----------------------------
build_sent_lstm <- function(timesteps, n_features) {
  keras_model_sequential() |>
    layer_lstm(units = 64, input_shape = c(timesteps, n_features)) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1)
}

build_sent_gru <- function(timesteps, n_features) {
  keras_model_sequential() |>
    layer_gru(units = 64, input_shape = c(timesteps, n_features)) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1)
}

build_sent_bi_lstm <- function(timesteps, n_features) {
  keras_model_sequential() |>
    bidirectional(layer_lstm(units = 64), input_shape = c(timesteps, n_features)) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1)
}

build_sent_bi_gru <- function(timesteps, n_features) {
  keras_model_sequential() |>
    bidirectional(layer_gru(units = 64), input_shape = c(timesteps, n_features)) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1)
}

build_sent_cnn_lstm <- function(timesteps, n_features) {
  keras_model_sequential() |>
    layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu",
                  input_shape = c(timesteps, n_features)) |>
    layer_max_pooling_1d(pool_size = 2) |>
    layer_lstm(units = 64) |>
    layer_dense(units = 1)
}

build_sent_cnn_bi_lstm <- function(timesteps, n_features) {
  keras_model_sequential() |>
    layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu",
                  input_shape = c(timesteps, n_features)) |>
    layer_max_pooling_1d(pool_size = 2) |>
    bidirectional(layer_lstm(units = 64)) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1)
}

build_sent_cnn_bi_lstm_attention <- function(timesteps, n_features) {
  inputs <- layer_input(shape = c(timesteps, n_features))

  x <- inputs |>
    layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu") |>
    layer_max_pooling_1d(pool_size = 2) |>
    bidirectional(layer_lstm(units = 64, return_sequences = TRUE))

  score <- x |>
    layer_dense(units = 1, activation = "tanh")

  attn_weights <- score |>
    layer_lambda(function(s) {
      e <- tf$squeeze(s, axis = -1L)
      a <- tf$nn$softmax(e, axis = -1L)
      tf$expand_dims(a, axis = -1L)
    })

  context <- layer_lambda(list(x, attn_weights), function(z) {
    x_seq <- z[[1]]
    a     <- z[[2]]
    tf$reduce_sum(x_seq * a, axis = 1L)
  })

  outputs <- context |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1)

  keras_model(inputs = inputs, outputs = outputs)
}

build_encoder_decoder_lstm <- function(timesteps, n_features, horizon) {
  inputs <- layer_input(shape = c(timesteps, n_features))
  encoded <- inputs |> layer_lstm(units = 64)
  dec_in  <- encoded |> layer_repeat_vector(horizon)
  decoded <- dec_in |> layer_lstm(units = 64)
  outputs <- decoded |> layer_dense(units = 1)
  keras_model(inputs, outputs)
}

build_encoder_decoder_gru <- function(timesteps, n_features, horizon) {
  inputs <- layer_input(shape = c(timesteps, n_features))
  encoded <- inputs |> layer_gru(units = 64)
  dec_in  <- encoded |> layer_repeat_vector(horizon)
  decoded <- dec_in |> layer_gru(units = 64)
  outputs <- decoded |> layer_dense(units = 1)
  keras_model(inputs, outputs)
}

get_model <- function(model_name, timesteps, n_features, horizon) {
  switch(model_name,
         "SENT-LSTM" = build_sent_lstm(timesteps, n_features),
         "SENT-GRU" = build_sent_gru(timesteps, n_features),
         "SENT-Bi-LSTM" = build_sent_bi_lstm(timesteps, n_features),
         "SENT-Bi-GRU" = build_sent_bi_gru(timesteps, n_features),
         "SENT-CNN-LSTM" = build_sent_cnn_lstm(timesteps, n_features),
         "SENT-CNN-Bi-LSTM" = build_sent_cnn_bi_lstm(timesteps, n_features),
         "SENT-CNN-Bi-LSTM-Attention" = build_sent_cnn_bi_lstm_attention(timesteps, n_features),
         "SENT-Encoder-decoder-LSTM" = build_encoder_decoder_lstm(timesteps, n_features, horizon),
         "SENT-USDX-Encoder-decoder-GRU" = build_encoder_decoder_gru(timesteps, n_features, horizon),
         stop("Unknown MODEL_NAME: ", model_name)
  )
}

# ----------------------------
# Load data
# ----------------------------
if (!file.exists(DATA_PATH)) {
  stop("Dataset not found at: ", DATA_PATH,
       "\nPlace your CSV at dataset/processed_data_best_corr_sentiment.csv or edit DATA_PATH.")
}

df <- read_csv(DATA_PATH, show_col_types = FALSE)
if ("Date" %in% names(df)) df <- df |> arrange(Date)

features <- get_features_for_model(MODEL_NAME)
target   <- "BRENT Close"

need_cols <- unique(c(features, target))
missing_cols <- setdiff(need_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "),
       "\nEdit get_features_for_model() or rename columns in the CSV.")
}

# ----------------------------
# Split + scale (train only)
# ----------------------------
splits <- split_chrono(df, 0.8, 0.1)
scaler <- minmax_fit(splits$train, cols = need_cols)

train_s <- minmax_transform(splits$train, cols = need_cols, scaler)
val_s   <- minmax_transform(splits$val,   cols = need_cols, scaler)
test_s  <- minmax_transform(splits$test,  cols = need_cols, scaler)

# ----------------------------
# Build sequences
# ----------------------------
train_seq <- make_sequences(train_s, features, target, LOOKBACK, HORIZON)
val_seq   <- make_sequences(val_s,   features, target, LOOKBACK, HORIZON)
test_seq  <- make_sequences(test_s,  features, target, LOOKBACK, HORIZON)

timesteps  <- dim(train_seq$X)[2]
n_features <- dim(train_seq$X)[3]

# ----------------------------
# Train model (with early stopping)
# ----------------------------
model <- get_model(MODEL_NAME, timesteps, n_features, HORIZON)

model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "mse",
  metrics = list("mae")
)

callbacks <- list(
  callback_early_stopping(monitor = "val_loss", patience = 5, restore_best_weights = TRUE),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 2)
)

history <- model |> fit(
  x = train_seq$X, y = train_seq$y,
  validation_data = list(val_seq$X, val_seq$y),
  epochs = EPOCHS,
  batch_size = BATCH_SIZE,
  callbacks = callbacks,
  verbose = 2
)

# ----------------------------
# Evaluate on TEST (best weights are already restored)
# ----------------------------
pred_test <- as.numeric(model |> predict(test_seq$X))
y_test    <- as.numeric(test_seq$y)

# Report metrics on ORIGINAL scale only (best-score style)
pred_orig <- minmax_inverse_one(pred_test, target, scaler)
y_orig    <- minmax_inverse_one(y_test,  target, scaler)
m_orig    <- metrics(y_orig, pred_orig)

# Also capture best validation loss/mae from training history (optional, useful)
h <- as.data.frame(history$metrics)
best_val_loss <- min(h$val_loss, na.rm = TRUE)
best_val_mae  <- if ("val_mae" %in% names(h)) min(h$val_mae, na.rm = TRUE) else NA

row_out <- data.frame(
  model = MODEL_NAME,
  features = paste(features, collapse = ","),
  lookback = LOOKBACK,
  horizon = HORIZON,
  best_val_loss = best_val_loss,
  best_val_mae_scaled = best_val_mae,
  MAE = unname(m_orig["MAE"]),
  MSE = unname(m_orig["MSE"]),
  RMSE = unname(m_orig["RMSE"])
)

if (!file.exists(BEST_SCORES_FILE)) {
  write.csv(row_out, BEST_SCORES_FILE, row.names = FALSE)
} else {
  # append
  old <- read_csv(BEST_SCORES_FILE, show_col_types = FALSE)
  out <- bind_rows(old, row_out)
  write.csv(out, BEST_SCORES_FILE, row.names = FALSE)
}

cat("\nSaved best-score row to: ", BEST_SCORES_FILE, "\n", sep = "")
print(row_out)
