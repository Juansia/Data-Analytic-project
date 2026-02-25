# gwo_ensemble.R
# R-only Grey Wolf Optimizer (GWO) for weighted ensemble of model predictions.
# This script expects per-model prediction CSVs produced by train_models.R.
#
# Recommended workflow:
# 1) Run train_models.R multiple times (different MODEL_NAME) to produce prediction files in outputs/
# 2) (Best) Ensure each model also saves validation predictions as outputs/pred_val_<MODEL>.csv
# 3) Run this script to learn ensemble weights using GWO on validation, then evaluate on test.
#
# If validation prediction files are NOT available, the script will fall back to using test predictions
# to fit weights (this is not ideal; use only if you have no validation preds).

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# ----------------------------
# CONFIG (edit these)
# ----------------------------
OUT_DIR <- "outputs"

# List the models you trained (must match filenames pred_*_<MODEL>.csv)
MODELS <- c(
  "SENT-LSTM",
  "SENT-GRU",
  "SENT-Bi-LSTM",
  "SENT-Bi-GRU",
  "SENT-CNN-Bi-LSTM",
  "SENT-CNN-Bi-LSTM-Attention",
  "SENT-Encoder-decoder-LSTM",
  "SENT-USDX-Encoder-decoder-GRU"
)

# Optimization objective on validation set:
# Choose one: "RMSE" or "MAE"
OBJECTIVE <- "RMSE"

# GWO hyperparameters
N_WOLVES <- 20
N_ITERS  <- 80
SEED     <- 42

# Whether to use ORIGINAL scale columns (recommended)
# train_models.R saves: y_true_orig, y_pred_orig (and also scaled columns)
USE_ORIGINAL_SCALE <- TRUE

# ----------------------------
# Helpers
# ----------------------------
set.seed(SEED)

rmse <- function(y, p) sqrt(mean((p - y)^2))
mae  <- function(y, p) mean(abs(p - y))

metric_fn <- function(name) {
  if (toupper(name) == "RMSE") return(rmse)
  if (toupper(name) == "MAE")  return(mae)
  stop("Unknown OBJECTIVE: ", name)
}

# Softmax -> weights on simplex (nonnegative, sum=1)
softmax <- function(z) {
  z <- z - max(z)
  e <- exp(z)
  e / sum(e)
}

# Read a set of prediction files into a matrix of predictions + true values
read_pred_set <- function(models, prefix) {
  # prefix: "pred_val_" or "pred_test_"
  paths <- file.path(OUT_DIR, paste0(prefix, models, ".csv"))
  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0) {
    return(list(ok = FALSE, missing = missing))
  }

  dfs <- lapply(paths, read_csv, show_col_types = FALSE)

  if (USE_ORIGINAL_SCALE) {
    y_col <- "y_true_orig"
    p_col <- "y_pred_orig"
  } else {
    y_col <- "y_true_scaled"
    p_col <- "y_pred_scaled"
  }

  # Consistency check
  y0 <- dfs[[1]][[y_col]]
  for (i in seq_along(dfs)) {
    if (!all(dfs[[i]][[y_col]] == y0)) {
      stop("True values differ across files. Ensure all models were evaluated on the same split/window.")
    }
  }

  P <- do.call(cbind, lapply(dfs, function(d) d[[p_col]]))
  colnames(P) <- models
  list(ok = TRUE, y = y0, P = P)
}

# Objective for weights
objective_for_weights <- function(w, y, P, metric) {
  p_ens <- as.numeric(P %*% w)
  metric(y, p_ens)
}

# ----------------------------
# Load validation and test prediction sets
# ----------------------------
val <- read_pred_set(MODELS, "pred_val_")
use_test_for_fit <- FALSE

if (!val$ok) {
  message("Validation prediction files not found. Missing:")
  message(paste0(" - ", val$missing, collapse = "\n"))
  message("\nFalling back to fitting weights on TEST predictions (not recommended).")
  use_test_for_fit <- TRUE
}

test <- read_pred_set(MODELS, "pred_test_")
if (!test$ok) {
  stop("Test prediction files missing. Please run train_models.R for each model first. Missing:\n",
       paste0(test$missing, collapse = "\n"))
}

fit_set <- if (use_test_for_fit) test else val
y_fit <- fit_set$y
P_fit <- fit_set$P

y_test <- test$y
P_test <- test$P

metric <- metric_fn(OBJECTIVE)

# ----------------------------
# Grey Wolf Optimizer (GWO) on simplex via softmax parameterization
# We optimize unconstrained vectors z; weights w = softmax(z)
# ----------------------------
D <- length(MODELS)

# Initialize wolves in unconstrained space
wolves <- matrix(rnorm(N_WOLVES * D, sd = 1), nrow = N_WOLVES, ncol = D)

eval_wolf <- function(z) {
  w <- softmax(z)
  objective_for_weights(w, y_fit, P_fit, metric)
}

fitness <- apply(wolves, 1, eval_wolf)

# Track bests (alpha, beta, delta)
order_idx <- order(fitness)
alpha <- wolves[order_idx[1], , drop = FALSE]
beta  <- wolves[order_idx[2], , drop = FALSE]
delta <- wolves[order_idx[3], , drop = FALSE]
alpha_fit <- fitness[order_idx[1]]

for (t in 1:N_ITERS) {
  a <- 2 - 2 * (t - 1) / (N_ITERS - 1)  # linearly decreases 2 -> 0

  for (i in 1:N_WOLVES) {
    # Update each dimension using alpha/beta/delta guidance
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

  fitness <- apply(wolves, 1, eval_wolf)
  order_idx <- order(fitness)

  alpha <- wolves[order_idx[1], , drop = FALSE]
  beta  <- wolves[order_idx[2], , drop = FALSE]
  delta <- wolves[order_idx[3], , drop = FALSE]
  alpha_fit <- fitness[order_idx[1]]

  if (t %% 10 == 0) {
    message(sprintf("Iter %d/%d | Best %s (fit set): %.6f", t, N_ITERS, OBJECTIVE, alpha_fit))
  }
}

best_w <- softmax(alpha[1, ])
names(best_w) <- MODELS

# ----------------------------
# Evaluate ensemble on TEST
# ----------------------------
p_ens_test <- as.numeric(P_test %*% best_w)

mae_test  <- mae(y_test, p_ens_test)
mse_test  <- mean((p_ens_test - y_test)^2)
rmse_test <- sqrt(mse_test)

ensemble_metrics <- data.frame(
  objective_fit_set = OBJECTIVE,
  used_test_for_fit = use_test_for_fit,
  MAE_test = mae_test,
  MSE_test = mse_test,
  RMSE_test = rmse_test
)

# Save outputs
weights_path <- file.path(OUT_DIR, "best_ensemble_weights_gwo.csv")
metrics_path <- file.path(OUT_DIR, "ensemble_metrics_gwo.csv")
preds_path   <- file.path(OUT_DIR, "ensemble_predictions_test_gwo.csv")

write.csv(data.frame(model = MODELS, weight = as.numeric(best_w)), weights_path, row.names = FALSE)
write.csv(ensemble_metrics, metrics_path, row.names = FALSE)

pred_df <- data.frame(
  y_true = y_test,
  y_pred_ensemble = p_ens_test
)
write.csv(pred_df, preds_path, row.names = FALSE)

cat("\nSaved files:\n",
    " - ", weights_path, "\n",
    " - ", metrics_path, "\n",
    " - ", preds_path, "\n", sep = "")

# Print weights
cat("\nBest weights (sum should be 1):\n")
print(round(best_w, 6))
cat("Sum weights:", sum(best_w), "\n")
