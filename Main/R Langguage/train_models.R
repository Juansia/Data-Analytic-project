# train_models.R
# ------------------------------------------------------------------------------
# Data-Driven Evaluation of Brent Crude Oil Forecasting Models (Project-aligned)
#
# - 3-day-ahead forecasting (HORIZON=3)
# - Chronological split: 80% train / 10% validation / 10% test
# - Min–max scaling fit on TRAIN ONLY (prevents leakage)
# - Sliding-window supervised learning with tunable Time Steps (TS) in [3, 30]
# - Grey Wolf Optimizer (GWO) hyperparameter search to minimize VALIDATION MSE
# - Saves per-model validation + test predictions for downstream ensemble weighting
#
# OPTIONAL: To mirror the PDF's reproducibility instructions (train one selected model at a time),
# set MODEL_NAME (e.g., "SENT-Bi-GRU") and run source("train_models.R").
# If MODEL_NAME is NULL, the script will tune/train ALL architectures listed in ARCHITECTURES.
#
# Outputs (created under OUT_DIR):
# - best_scores_r.csv (append one row per trained model: metrics + best hyperparams)
# - pred_val_<MODEL>.csv  (validation predictions with target_date)
# - pred_test_<MODEL>.csv (test predictions with target_date)
# - best_params_<ARCH>.csv (best hyperparams per architecture)
# - gwo_history_<ARCH>.csv (best validation MSE trace over iterations)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(keras)
  library(tensorflow)
})

# ----------------------------
# CONFIG (EDIT THESE)
# ----------------------------

# Path to your final processed CSV (as described in the PDF: saved under Main/R Language/dataset/)
DATA_PATH <- "processed_data_best_corr_sentiment.csv"  # <-- change if needed
OUT_DIR   <- "outputs"

# Column names in your processed CSV (edit if your headers differ)
COL_DATE  <- NULL                 # if NULL, auto-detect 'date' or 'Date'
COL_BRENT <- "BRENT Close"        # Brent Close column
COL_USDX  <- "USDX"               # USDX column
COL_SENT  <- "SENT"               # Sentiment column (you renamed Csum_CrudeBERT_Plus_GT -> SENT)

# Forecasting setup (paper)
HORIZON <- 3

# Time split (paper)
TRAIN_FRAC <- 0.80
VAL_FRAC   <- 0.10

# OPTIONAL: Train/tune a single model name (PDF suggests doing this iteratively).
# Examples:
#   "SENT-Bi-GRU"
#   "SENT-CNN-Bi-LSTM-Attention"
#   "SENT-USDX-Encoder-decoder-GRU"
MODEL_NAME <- NULL

# Architectures to tune/train (paper set)
ARCHITECTURES <- c(
  "LSTM",
  "GRU",
  "Bi-LSTM",
  "Bi-GRU",
  "CNN-LSTM",
  "CNN-Bi-LSTM",
  "CNN-Bi-LSTM-Attention",
  "Encoder-decoder-LSTM",
  "Encoder-decoder-GRU"
)

# Search space (Table 3)
FEATURE_SETS <- c("NONE", "USDX", "SENT", "USDX+SENT")
OPTIMIZERS   <- c("SGD", "RMSprop", "Adagrad", "Adadelta", "AdamW", "Adam", "Adamax")
HIDDEN_UNITS <- 2^(1:8)  # 2,4,...,256  (matches Table 3)

# GWO settings (not specified in the PDF; adjust for runtime vs quality)
N_WOLVES <- 12
N_ITERS  <- 24

# Training settings (Appendix Table 7 states: "All models trained for 20 epochs.")
EPOCHS_TUNE  <- 20
EPOCHS_FINAL <- 20
BATCH_SIZE   <- 32
SEED         <- 42

# The PDF does not mention early stopping; keep it OFF by default to match.
USE_EARLY_STOPPING_FINAL <- FALSE
ES_PATIENCE_FINAL        <- 25  # used only if USE_EARLY_STOPPING_FINAL=TRUE

# Save trained models (optional)
SAVE_MODELS <- TRUE
MODEL_DIR   <- file.path(OUT_DIR, "models")

# ----------------------------
# Derived (from MODEL_NAME)
# ----------------------------
FORCE_FEATURE_SET <- NULL  # if MODEL_NAME is set, we pin feature set to match the prefix
FORCE_ARCH        <- NULL  # if MODEL_NAME is set, we pin architecture to match the suffix

parse_model_name <- function(model_name) {
  # Accept prefixes:
  #   None-<ARCH>, USDX-<ARCH>, SENT-<ARCH>, SENT-USDX-<ARCH>
  # Map:
  #   None      -> NONE
  #   USDX      -> USDX
  #   SENT      -> SENT
  #   SENT-USDX -> USDX+SENT
  model_name <- trimws(model_name)

  if (startsWith(model_name, "SENT-USDX-")) {
    arch <- sub("^SENT-USDX-", "", model_name)
    return(list(feature_set = "USDX+SENT", arch = arch))
  }
  if (startsWith(model_name, "SENT-")) {
    arch <- sub("^SENT-", "", model_name)
    return(list(feature_set = "SENT", arch = arch))
  }
  if (startsWith(model_name, "USDX-")) {
    arch <- sub("^USDX-", "", model_name)
    return(list(feature_set = "USDX", arch = arch))
  }
  if (startsWith(model_name, "None-") || startsWith(model_name, "NONE-")) {
    arch <- sub("^(None|NONE)-", "", model_name)
    return(list(feature_set = "NONE", arch = arch))
  }

  stop("MODEL_NAME must start with one of: None-, USDX-, SENT-, SENT-USDX-. ",
       "Got: ", model_name,
       "\nExamples: 'SENT-Bi-GRU', 'SENT-USDX-Encoder-decoder-GRU'.")
}

if (!is.null(MODEL_NAME)) {
  sel <- parse_model_name(MODEL_NAME)
  FORCE_FEATURE_SET <- sel$feature_set
  FORCE_ARCH <- sel$arch

  if (!(FORCE_ARCH %in% ARCHITECTURES)) {
    stop("MODEL_NAME architecture '", FORCE_ARCH, "' is not in ARCHITECTURES.\n",
         "Available: ", paste(ARCHITECTURES, collapse = ", "))
  }

  ARCHITECTURES <- c(FORCE_ARCH)
  cat("MODE: single-model run\n")
  cat(" - MODEL_NAME:", MODEL_NAME, "\n")
  cat(" - FORCE_FEATURE_SET:", FORCE_FEATURE_SET, "\n")
  cat(" - FORCE_ARCH:", FORCE_ARCH, "\n")
} else {
  cat("MODE: full run (all ARCHITECTURES)\n")
}

# ----------------------------
# Setup output folders
# ----------------------------
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)

BEST_SCORES_FILE <- file.path(OUT_DIR, "best_scores_r.csv")

set.seed(SEED)
tensorflow::tf$random$set_seed(SEED)

# ----------------------------
# Helpers: scaling, metrics, sequences
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

minmax_inverse_vec <- function(x, col, scaler) {
  x * (scaler$max[[col]] - scaler$min[[col]] + 1e-9) + scaler$min[[col]]
}

metrics <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  mae  <- mean(abs(y_pred - y_true))
  mse  <- mean((y_pred - y_true)^2)
  rmse <- sqrt(mse)
  c(MAE = mae, MSE = mse, RMSE = rmse)
}

split_indices <- function(n, train_frac, val_frac) {
  i_train <- floor(train_frac * n)
  i_val   <- floor((train_frac + val_frac) * n)
  list(i_train = i_train, i_val = i_val)
}

# Make sliding-window sequences across the FULL series and then select by TARGET index range.
# This avoids losing the early validation/test targets due to split boundaries.
make_sequences_by_target_range <- function(df_scaled, dates,
                                          feature_cols, target_col,
                                          lookback, horizon,
                                          target_idx_min, target_idx_max) {
  n <- nrow(df_scaled)
  last_start <- n - lookback - horizon + 1
  if (last_start < 1) stop("Not enough rows for lookback + horizon. Reduce TS or add data.")

  target_idx_all <- (1:last_start) + lookback + horizon - 1
  keep <- which(target_idx_all >= target_idx_min & target_idx_all <= target_idx_max)

  m <- length(keep)
  if (m < 5) stop("Too few samples in this split after sequencing (m=", m, "). Try smaller TS.")

  n_features <- length(feature_cols)
  X <- array(0, dim = c(m, lookback, n_features))
  y <- array(0, dim = c(m, 1))
  t_dates <- dates[target_idx_all[keep]]

  for (k in seq_along(keep)) {
    i <- keep[k]
    X[k, , ] <- as.matrix(df_scaled[i:(i + lookback - 1), feature_cols, drop = FALSE])
    y[k, 1]  <- df_scaled[[target_col]][i + lookback + horizon - 1]
  }

  list(X = X, y = y, target_date = t_dates, target_index = target_idx_all[keep])
}

feature_cols_for_set <- function(feature_set) {
  feature_set <- toupper(feature_set)
  base <- c(COL_BRENT)
  if (feature_set == "NONE") return(base)
  if (feature_set == "USDX") return(c(base, COL_USDX))
  if (feature_set == "SENT") return(c(base, COL_SENT))
  if (feature_set == "USDX+SENT") return(c(base, COL_USDX, COL_SENT))
  stop("Unknown FEATURE_SET: ", feature_set)
}

feature_prefix_for_set <- function(feature_set) {
  feature_set <- toupper(feature_set)
  if (feature_set == "NONE") return("None")
  if (feature_set == "USDX") return("USDX")
  if (feature_set == "SENT") return("SENT")
  if (feature_set == "USDX+SENT") return("SENT-USDX")  # matches the PDF's naming
  stop("Unknown FEATURE_SET: ", feature_set)
}

# ----------------------------
# Optimizer factory (robust AdamW)
# ----------------------------
make_optimizer <- function(name, lr) {
  nm <- toupper(name)

  if (nm == "SGD")      return(optimizer_sgd(learning_rate = lr))
  if (nm == "RMSPROP")  return(optimizer_rmsprop(learning_rate = lr))
  if (nm == "ADAGRAD")  return(optimizer_adagrad(learning_rate = lr))
  if (nm == "ADADELTA") return(optimizer_adadelta(learning_rate = lr))
  if (nm == "ADAM")     return(optimizer_adam(learning_rate = lr))
  if (nm == "ADAMAX")   return(optimizer_adamax(learning_rate = lr))

  if (nm == "ADAMW") {
    # Try tf.keras AdamW (available in many TF versions).
    opt <- tryCatch({
      tensorflow::tf$keras$optimizers$AdamW(learning_rate = lr, weight_decay = 1e-4)
    }, error = function(e) NULL)
    if (!is.null(opt)) return(opt)

    # Fallback: experimental namespace in some TF builds
    opt <- tryCatch({
      tensorflow::tf$keras$optimizers$experimental$AdamW(learning_rate = lr, weight_decay = 1e-4)
    }, error = function(e) NULL)
    if (!is.null(opt)) return(opt)

    message("WARNING: AdamW not available in this TF build; falling back to Adam.")
    return(optimizer_adam(learning_rate = lr))
  }

  stop("Unknown optimizer: ", name)
}

# ----------------------------
# Model builders (parameterized by hidden units + dropout)
# ----------------------------
build_model <- function(arch, timesteps, n_features, units, dropout, horizon) {
  arch <- as.character(arch)

  if (arch == "LSTM") {
    model <- keras_model_sequential() |>
      layer_lstm(units = units, input_shape = c(timesteps, n_features)) |>
      layer_dropout(rate = dropout) |>
      layer_dense(units = 1)
    return(model)
  }

  if (arch == "GRU") {
    model <- keras_model_sequential() |>
      layer_gru(units = units, input_shape = c(timesteps, n_features)) |>
      layer_dropout(rate = dropout) |>
      layer_dense(units = 1)
    return(model)
  }

  # NOTE: Put input_shape on the wrapped layer (more robust across keras versions)
  if (arch == "Bi-LSTM") {
    model <- keras_model_sequential() |>
      bidirectional(layer_lstm(units = units, input_shape = c(timesteps, n_features))) |>
      layer_dropout(rate = dropout) |>
      layer_dense(units = 1)
    return(model)
  }

  if (arch == "Bi-GRU") {
    model <- keras_model_sequential() |>
      bidirectional(layer_gru(units = units, input_shape = c(timesteps, n_features))) |>
      layer_dropout(rate = dropout) |>
      layer_dense(units = 1)
    return(model)
  }

  if (arch == "CNN-LSTM") {
    model <- keras_model_sequential() |>
      layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu",
                    input_shape = c(timesteps, n_features)) |>
      layer_max_pooling_1d(pool_size = 2) |>
      layer_lstm(units = units) |>
      layer_dropout(rate = dropout) |>
      layer_dense(units = 1)
    return(model)
  }

  if (arch == "CNN-Bi-LSTM") {
    model <- keras_model_sequential() |>
      layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu",
                    input_shape = c(timesteps, n_features)) |>
      layer_max_pooling_1d(pool_size = 2) |>
      bidirectional(layer_lstm(units = units)) |>
      layer_dropout(rate = dropout) |>
      layer_dense(units = 1)
    return(model)
  }

  if (arch == "CNN-Bi-LSTM-Attention") {
    inputs <- layer_input(shape = c(timesteps, n_features))

    x <- inputs |>
      layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu") |>
      layer_max_pooling_1d(pool_size = 2) |>
      bidirectional(layer_lstm(units = units, return_sequences = TRUE)) |>
      layer_dropout(rate = dropout)

    # Simple learned attention over timesteps
    score <- x |> layer_dense(units = 1, activation = "tanh")

    attn_weights <- score |>
      layer_lambda(function(s) {
        e <- tensorflow::tf$squeeze(s, axis = -1L)
        a <- tensorflow::tf$nn$softmax(e, axis = -1L)
        tensorflow::tf$expand_dims(a, axis = -1L)
      })

    context <- layer_lambda(list(x, attn_weights), function(z) {
      x_seq <- z[[1]]
      a     <- z[[2]]
      tensorflow::tf$reduce_sum(x_seq * a, axis = 1L)
    })

    outputs <- context |> layer_dense(units = 1)
    model <- keras_model(inputs = inputs, outputs = outputs)
    return(model)
  }

  if (arch == "Encoder-decoder-LSTM") {
    inputs  <- layer_input(shape = c(timesteps, n_features))
    encoded <- inputs |> layer_lstm(units = units)
    encoded <- encoded |> layer_dropout(rate = dropout)
    dec_in  <- encoded |> layer_repeat_vector(horizon)
    decoded <- dec_in |> layer_lstm(units = units)
    outputs <- decoded |> layer_dense(units = 1)
    model <- keras_model(inputs, outputs)
    return(model)
  }

  if (arch == "Encoder-decoder-GRU") {
    inputs  <- layer_input(shape = c(timesteps, n_features))
    encoded <- inputs |> layer_gru(units = units)
    encoded <- encoded |> layer_dropout(rate = dropout)
    dec_in  <- encoded |> layer_repeat_vector(horizon)
    decoded <- dec_in |> layer_gru(units = units)
    outputs <- decoded |> layer_dense(units = 1)
    model <- keras_model(inputs, outputs)
    return(model)
  }

  stop("Unknown architecture: ", arch)
}

# ----------------------------
# GWO: decode wolf positions -> hyperparameters
# (Mixed discrete/continuous variables via normalized [0,1] positions)
# ----------------------------
decode_wolf <- function(pos) {
  # pos in [0,1] of length 6:
  # 1) lr_u, 2) hidden_u, 3) dropout_u, 4) ts_u, 5) opt_u, 6) feat_u
  lr_u     <- pos[1]
  hidden_u <- pos[2]
  drop_u   <- pos[3]
  ts_u     <- pos[4]
  opt_u    <- pos[5]
  feat_u   <- pos[6]

  # Learning rate: log-uniform in [1e-4, 1e-1]
  lr <- 10^(log10(1e-4) + lr_u * (log10(1e-1) - log10(1e-4)))

  # Hidden units: pick from powers-of-two list (Table 3)
  idx_h <- floor(hidden_u * length(HIDDEN_UNITS)) + 1
  idx_h <- max(1, min(length(HIDDEN_UNITS), idx_h))
  units <- HIDDEN_UNITS[idx_h]

  # Dropout: uniform [0.2, 0.5]
  dropout <- 0.2 + drop_u * (0.5 - 0.2)

  # Time steps: integer [3, 30]
  ts <- round(3 + ts_u * (30 - 3))
  ts <- max(3, min(30, ts))

  # Optimizer: categorical
  idx_o <- floor(opt_u * length(OPTIMIZERS)) + 1
  idx_o <- max(1, min(length(OPTIMIZERS), idx_o))
  opt <- OPTIMIZERS[idx_o]

  # Feature set: categorical, unless forced by MODEL_NAME mode
  if (!is.null(FORCE_FEATURE_SET)) {
    feat <- FORCE_FEATURE_SET
  } else {
    idx_f <- floor(feat_u * length(FEATURE_SETS)) + 1
    idx_f <- max(1, min(length(FEATURE_SETS), idx_f))
    feat <- FEATURE_SETS[idx_f]
  }

  list(
    lr = lr,
    units = units,
    dropout = dropout,
    timesteps = ts,
    optimizer = opt,
    feature_set = feat
  )
}

# ----------------------------
# Load + prepare data
# ----------------------------
if (!file.exists(DATA_PATH)) {
  stop("Dataset not found at: ", DATA_PATH, "\nEdit DATA_PATH at the top of this file.")
}

df <- read_csv(DATA_PATH, show_col_types = FALSE)

# Identify date column
date_col <- COL_DATE
if (is.null(date_col)) {
  if ("date" %in% names(df)) date_col <- "date"
  if ("Date" %in% names(df)) date_col <- "Date"
}
if (is.null(date_col)) stop("No date column found. Expected a 'date' or 'Date' column, or set COL_DATE.")

# Sort by date (chronological)
df[[date_col]] <- as.character(df[[date_col]])
date_parsed <- suppressWarnings(as.Date(df[[date_col]]))
if (all(!is.na(date_parsed))) {
  df[[date_col]] <- date_parsed
  df <- df |> arrange(.data[[date_col]])
} else {
  df <- df |> arrange(.data[[date_col]])
}

# Required cols for this project
cols_all <- c(COL_BRENT, COL_USDX, COL_SENT)
missing_cols <- setdiff(cols_all, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns in CSV: ", paste(missing_cols, collapse = ", "),
       "\nExpected at least: ", paste(cols_all, collapse = ", "),
       "\nEdit COL_BRENT/COL_USDX/COL_SENT at the top if your headers differ.")
}

# Drop rows with NA in required cols
df <- df[complete.cases(df[, cols_all]), ]

n <- nrow(df)
idx <- split_indices(n, TRAIN_FRAC, VAL_FRAC)
i_train <- idx$i_train
i_val   <- idx$i_val

cat("Rows:", n, "\nTrain end idx:", i_train, "\nVal end idx:", i_val, "\nTest end idx:", n, "\n")

# Fit scaler ONLY on train portion (paper requirement)
scaler <- minmax_fit(df[1:i_train, ], cols_all)

# Scale full dataset using train scaler
df_s <- minmax_transform(df, cols_all, scaler)

dates <- df[[date_col]]
TARGET_COL <- COL_BRENT

# ----------------------------
# Core evaluation: train config -> validation MSE (original scale)
# ----------------------------
eval_config <- function(arch, cfg) {
  # Fast reject invalid CNN settings (conv+pool needs enough timesteps)
  # This avoids repeated shape errors during the GWO search.
  if (grepl("^CNN", arch) && cfg$timesteps < 6) {
    return(list(val_mse = Inf))
  }

  features <- feature_cols_for_set(cfg$feature_set)
  lookback <- cfg$timesteps

  train_seq <- make_sequences_by_target_range(
    df_s, dates, features, TARGET_COL, lookback, HORIZON,
    target_idx_min = 1, target_idx_max = i_train
  )

  val_seq <- make_sequences_by_target_range(
    df_s, dates, features, TARGET_COL, lookback, HORIZON,
    target_idx_min = i_train + 1, target_idx_max = i_val
  )

  tryCatch({ keras::k_clear_session() }, error = function(e) NULL)
  gc()

  set.seed(SEED)
  tensorflow::tf$random$set_seed(SEED)

  model <- build_model(arch, lookback, length(features), cfg$units, cfg$dropout, HORIZON)

  model |> compile(
    optimizer = make_optimizer(cfg$optimizer, cfg$lr),
    loss      = "mse"
  )

  model |> fit(
    x = train_seq$X, y = train_seq$y,
    validation_data = list(val_seq$X, val_seq$y),
    epochs = EPOCHS_TUNE,
    batch_size = BATCH_SIZE,
    verbose = 0
  )

  pred_val_scaled <- as.numeric(model |> predict(val_seq$X, verbose = 0))
  y_val_scaled    <- as.numeric(val_seq$y)

  pred_val_orig <- minmax_inverse_vec(pred_val_scaled, TARGET_COL, scaler)
  y_val_orig    <- minmax_inverse_vec(y_val_scaled,    TARGET_COL, scaler)

  val_mse <- mean((pred_val_orig - y_val_orig)^2)
  if (!is.finite(val_mse)) val_mse <- Inf

  list(val_mse = val_mse)
}

# ----------------------------
# Grey Wolf Optimizer (GWO) for one architecture
# ----------------------------
gwo_optimize_arch <- function(arch) {
  cat("\n==============================\n")
  cat("GWO tuning architecture:", arch, "\n")
  cat("Wolves:", N_WOLVES, " | Iters:", N_ITERS, " | Tune epochs:", EPOCHS_TUNE, "\n")
  if (!is.null(FORCE_FEATURE_SET)) cat("Feature set fixed to:", FORCE_FEATURE_SET, "\n")
  cat("==============================\n")

  D <- 6  # lr, hidden, dropout, ts, opt, feat

  wolves <- matrix(runif(N_WOLVES * D), nrow = N_WOLVES, ncol = D)

  eval_pos <- function(pos) {
    cfg <- decode_wolf(pos)
    fit <- tryCatch({
      eval_config(arch, cfg)$val_mse
    }, error = function(e) {
      message("WARN eval failed: ", conditionMessage(e))
      Inf
    })
    fit
  }

  fitness <- apply(wolves, 1, eval_pos)

  ord <- order(fitness)
  alpha <- wolves[ord[1], , drop = FALSE]
  beta  <- wolves[ord[2], , drop = FALSE]
  delta <- wolves[ord[3], , drop = FALSE]
  alpha_fit <- fitness[ord[1]]

  best_hist <- data.frame(iter = 0, best_val_mse = alpha_fit)

  # Track global best across iterations (GWO fitness is not guaranteed monotonic)
  best_so_far <- alpha_fit
  best_alpha  <- alpha

  for (t in 1:N_ITERS) {
    a <- 2 - 2 * (t - 1) / max(1, (N_ITERS - 1))

    for (i in 1:N_WOLVES) {
      for (j in 1:D) {
        r1 <- runif(1); r2 <- runif(1)
        A1 <- 2 * a * r1 - a
        C1 <- 2 * r2
        D_alpha <- abs(C1 * alpha[1, j] - wolves[i, j])
        X1 <- alpha[1, j] - A1 * D_alpha

        r1 <- runif(1); r2 <- runif(1)
        A2 <- 2 * a * r1 - a
        C2 <- 2 * r2
        D_beta <- abs(C2 * beta[1, j] - wolves[i, j])
        X2 <- beta[1, j] - A2 * D_beta

        r1 <- runif(1); r2 <- runif(1)
        A3 <- 2 * a * r1 - a
        C3 <- 2 * r2
        D_delta <- abs(C3 * delta[1, j] - wolves[i, j])
        X3 <- delta[1, j] - A3 * D_delta

        wolves[i, j] <- (X1 + X2 + X3) / 3
      }
    }

    wolves <- pmin(pmax(wolves, 0), 1)

    fitness <- apply(wolves, 1, eval_pos)
    ord <- order(fitness)

    alpha <- wolves[ord[1], , drop = FALSE]
    beta  <- wolves[ord[2], , drop = FALSE]
    delta <- wolves[ord[3], , drop = FALSE]
    alpha_fit <- fitness[ord[1]]

    if (alpha_fit + 1e-12 < best_so_far) {
      best_so_far <- alpha_fit
      best_alpha  <- alpha
    }

    best_hist <- bind_rows(best_hist, data.frame(iter = t, best_val_mse = alpha_fit))

    if (t %% 2 == 0) {
      cat(sprintf("Iter %d/%d | best val MSE: %.6f\n", t, N_ITERS, alpha_fit))
    }
  }

  best_cfg <- decode_wolf(best_alpha[1, ])
  list(best_cfg = best_cfg, best_val_mse = best_so_far, history = best_hist)
}

# ----------------------------
# Train final model and save predictions/metrics
# ----------------------------
train_and_save_best <- function(arch, cfg, tuned_val_mse) {
  features <- feature_cols_for_set(cfg$feature_set)
  lookback <- cfg$timesteps

  train_seq <- make_sequences_by_target_range(
    df_s, dates, features, TARGET_COL, lookback, HORIZON,
    target_idx_min = 1, target_idx_max = i_train
  )
  val_seq <- make_sequences_by_target_range(
    df_s, dates, features, TARGET_COL, lookback, HORIZON,
    target_idx_min = i_train + 1, target_idx_max = i_val
  )
  test_seq <- make_sequences_by_target_range(
    df_s, dates, features, TARGET_COL, lookback, HORIZON,
    target_idx_min = i_val + 1, target_idx_max = n
  )

  tryCatch({ keras::k_clear_session() }, error = function(e) NULL)
  gc()

  set.seed(SEED)
  tensorflow::tf$random$set_seed(SEED)

  model <- build_model(arch, lookback, length(features), cfg$units, cfg$dropout, HORIZON)

  model |> compile(
    optimizer = make_optimizer(cfg$optimizer, cfg$lr),
    loss      = "mse",
    metrics   = list("mae")
  )

  callbacks <- list()
  if (USE_EARLY_STOPPING_FINAL) {
    callbacks <- list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = ES_PATIENCE_FINAL,
        restore_best_weights = TRUE
      )
    )
  }

  cat("\nFinal training:", arch, " | Feature set:", cfg$feature_set,
      " | TS:", lookback, " | H:", cfg$units,
      " | Drop:", sprintf("%.3f", cfg$dropout),
      " | LR:", sprintf("%.6f", cfg$lr),
      " | Opt:", cfg$optimizer, "\n", sep = "")

  model |> fit(
    x = train_seq$X, y = train_seq$y,
    validation_data = list(val_seq$X, val_seq$y),
    epochs = EPOCHS_FINAL,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    verbose = 2
  )

  # Predictions (val + test)
  pred_val_scaled  <- as.numeric(model |> predict(val_seq$X, verbose = 0))
  pred_test_scaled <- as.numeric(model |> predict(test_seq$X, verbose = 0))
  y_val_scaled     <- as.numeric(val_seq$y)
  y_test_scaled    <- as.numeric(test_seq$y)

  pred_val_orig  <- minmax_inverse_vec(pred_val_scaled,  TARGET_COL, scaler)
  pred_test_orig <- minmax_inverse_vec(pred_test_scaled, TARGET_COL, scaler)
  y_val_orig     <- minmax_inverse_vec(y_val_scaled,     TARGET_COL, scaler)
  y_test_orig    <- minmax_inverse_vec(y_test_scaled,    TARGET_COL, scaler)

  m_val  <- metrics(y_val_orig,  pred_val_orig)
  m_test <- metrics(y_test_orig, pred_test_orig)

  # Naming
  feature_prefix <- feature_prefix_for_set(cfg$feature_set)
  model_id <- paste0(feature_prefix, "-", arch)

  # Save model (optional)
  if (SAVE_MODELS) {
    model_path <- file.path(MODEL_DIR, paste0(model_id, ".keras"))
    tryCatch({
      save_model_tf(model, model_path)
      cat("Saved model to: ", model_path, "\n", sep = "")
    }, error = function(e) {
      message("WARN: could not save model: ", conditionMessage(e))
    })
  }

  # Save prediction CSVs with target_date for alignment in ensemble
  val_out <- data.frame(
    target_date   = as.character(val_seq$target_date),
    target_index  = val_seq$target_index,
    y_true_orig   = y_val_orig,
    y_pred_orig   = pred_val_orig
  )
  test_out <- data.frame(
    target_date   = as.character(test_seq$target_date),
    target_index  = test_seq$target_index,
    y_true_orig   = y_test_orig,
    y_pred_orig   = pred_test_orig
  )

  write.csv(val_out,  file.path(OUT_DIR, paste0("pred_val_",  model_id, ".csv")), row.names = FALSE)
  write.csv(test_out, file.path(OUT_DIR, paste0("pred_test_", model_id, ".csv")), row.names = FALSE)

  # Save best params per arch
  params_row <- data.frame(
    architecture = arch,
    model_id = model_id,
    feature_set = cfg$feature_set,
    timesteps = cfg$timesteps,
    lr = cfg$lr,
    dropout = cfg$dropout,
    hidden_units = cfg$units,
    optimizer = cfg$optimizer,
    tuned_val_mse = tuned_val_mse,
    val_MAE = unname(m_val["MAE"]),
    val_MSE = unname(m_val["MSE"]),
    val_RMSE = unname(m_val["RMSE"]),
    test_MAE = unname(m_test["MAE"]),
    test_MSE = unname(m_test["MSE"]),
    test_RMSE = unname(m_test["RMSE"])
  )
  write.csv(params_row, file.path(OUT_DIR, paste0("best_params_", arch, ".csv")), row.names = FALSE)

  # Append to global best score file
  if (!file.exists(BEST_SCORES_FILE)) {
    write.csv(params_row, BEST_SCORES_FILE, row.names = FALSE)
  } else {
    old <- read_csv(BEST_SCORES_FILE, show_col_types = FALSE)
    out <- bind_rows(old, params_row)
    write.csv(out, BEST_SCORES_FILE, row.names = FALSE)
  }

  cat("Saved predictions:\n",
      " - ", file.path(OUT_DIR, paste0("pred_val_",  model_id, ".csv")), "\n",
      " - ", file.path(OUT_DIR, paste0("pred_test_", model_id, ".csv")), "\n", sep = "")
  cat("Saved params + scores:\n",
      " - ", file.path(OUT_DIR, paste0("best_params_", arch, ".csv")), "\n",
      " - ", BEST_SCORES_FILE, "\n", sep = "")

  invisible(params_row)
}

# ----------------------------
# MAIN: tune + train per architecture
# ----------------------------
cat("\n=== TRAIN MODELS (GWO) ===\n")
cat("HORIZON:", HORIZON, "\n")
cat("Train/Val/Test:", TRAIN_FRAC, "/", VAL_FRAC, "/", (1 - TRAIN_FRAC - VAL_FRAC), "\n")
cat("Epochs (tune/final):", EPOCHS_TUNE, "/", EPOCHS_FINAL, "\n")

for (arch in ARCHITECTURES) {
  res <- gwo_optimize_arch(arch)

  # Save tuning trace (best validation MSE per iteration)
  tryCatch({
    write.csv(res$history, file.path(OUT_DIR, paste0("gwo_history_", arch, ".csv")), row.names = FALSE)
  }, error = function(e) {
    message("WARN: could not write GWO history: ", conditionMessage(e))
  })

  best_cfg <- res$best_cfg
  cat("\nBest for ", arch, ":\n", sep = "")
  print(best_cfg)
  cat("Best tuned validation MSE (orig scale):", res$best_val_mse, "\n")

  train_and_save_best(arch, best_cfg, tuned_val_mse = res$best_val_mse)
}

cat("\nDONE.\nNext: run gwo_ensemble_v2_fixed.R to learn ensemble weights.\n")
