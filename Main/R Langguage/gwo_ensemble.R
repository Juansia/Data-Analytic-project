# gwo_ensemble.R
# ------------------------------------------------------------------------------
# Grey Wolf Optimizer (GWO) for weighted ensemble of model predictions.
#
# This script expects the prediction CSVs produced by train_models.R:
#   outputs/pred_val_<MODEL>.csv
#   outputs/pred_test_<MODEL>.csv
#
# Each prediction file must include:
#   target_date, y_true_orig, y_pred_orig
#
# Key improvement vs older versions:
# - Aligns predictions by target_date (works even when models use different TS/lookback)
# - Fits weights ONLY on validation by default (prevents leakage)
# - Enforces non-negative weights summing to 1 via softmax(z)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# ----------------------------
# CONFIG (EDIT THESE)
# ----------------------------
OUT_DIR <- "outputs"

# If NULL: auto-detect models from outputs/pred_val_*.csv
# If set: must match filename suffix exactly (e.g., "SENT-Bi-GRU")
MODELS <- NULL

# Fit objective on validation set (project text: minimize validation MSE)
# Choose: "MSE", "RMSE", "MAE"
OBJECTIVE <- "MSE"

# GWO hyperparameters (weight search is cheap, so you can set these high)
N_WOLVES <- 40
N_ITERS  <- 250
SEED     <- 42

# Safety: do NOT fit on test unless you explicitly allow it
ALLOW_TEST_FIT_IF_NO_VAL <- FALSE

# ----------------------------
# Helpers
# ----------------------------
set.seed(SEED)

mse  <- function(y, p) mean((p - y)^2)
rmse <- function(y, p) sqrt(mean((p - y)^2))
mae  <- function(y, p) mean(abs(p - y))

metric_fn <- function(name) {
  nm <- toupper(name)
  if (nm == "MSE")  return(mse)
  if (nm == "RMSE") return(rmse)
  if (nm == "MAE")  return(mae)
  stop("Unknown OBJECTIVE: ", name)
}

softmax <- function(z) {
  z <- z - max(z)
  e <- exp(z)
  e / sum(e)
}

# Read prediction file for one model and split
read_pred <- function(model_id, split) {
  path <- file.path(OUT_DIR, paste0("pred_", split, "_", model_id, ".csv"))
  if (!file.exists(path)) stop("Missing file: ", path)

  df <- read_csv(path, show_col_types = FALSE)

  need <- c("target_date", "y_true_orig", "y_pred_orig")
  miss <- setdiff(need, names(df))
  if (length(miss) > 0) {
    stop("File ", path, " is missing required columns: ", paste(miss, collapse = ", "))
  }

  df |> transmute(
    target_date = as.character(target_date),
    y_true = as.numeric(y_true_orig),
    pred   = as.numeric(y_pred_orig)
  )
}

# Build aligned (date-joined) matrix for a split
build_matrix <- function(models, split) {
  dfs <- lapply(models, function(m) {
    d <- read_pred(m, split)
    names(d)[names(d) == "pred"] <- m
    d
  })

  # Reduce inner joins on target_date
  merged <- dfs[[1]]
  for (i in 2:length(dfs)) {
    merged <- merged |> inner_join(dfs[[i]], by = c("target_date", "y_true"))
  }

  # If y_true differs numerically due to floating, fall back to date-join then check
  if (nrow(merged) == 0) {
    merged <- dfs[[1]]
    for (i in 2:length(dfs)) {
      merged <- merged |> inner_join(dfs[[i]] |> select(-y_true), by = "target_date")
    }
    # Verify y_true consistency
    # Use y_true from the first model file, and ensure other files match within tolerance.
    for (m in models) {
      # We don't have other y_true columns anymore, so rebuild checks quickly
      other <- read_pred(m, split)
      chk <- merged |> inner_join(other, by = "target_date", suffix = c("", ".other"))
      if (max(abs(chk$y_true - chk$y_true.other), na.rm = TRUE) > 1e-6) {
        stop("y_true differs across models after aligning by date. Ensure all models were built from the same dataset & horizon.")
      }
    }
  }

  y <- merged$y_true
  P <- as.matrix(merged[, models, drop = FALSE])

  list(df = merged, y = y, P = P)
}

# ----------------------------
# Model discovery (if MODELS is NULL)
# ----------------------------
if (is.null(MODELS)) {
  files <- list.files(OUT_DIR, pattern = "^pred_val_.*\\.csv$", full.names = FALSE)
  if (length(files) == 0) stop("No validation prediction files found in ", OUT_DIR, ". Run train_models.R first.")

  MODELS <- sub("^pred_val_", "", files)
  MODELS <- sub("\\.csv$", "", MODELS)

  # Keep only those that also have test files
  MODELS <- MODELS[file.exists(file.path(OUT_DIR, paste0("pred_test_", MODELS, ".csv")))]

  if (length(MODELS) == 0) stop("No models have both pred_val_ and pred_test_ files.")
}

cat("Models in ensemble (", length(MODELS), "):\n", paste0(" - ", MODELS, collapse = "\n"), "\n", sep = "")

metric <- metric_fn(OBJECTIVE)

# ----------------------------
# Load aligned validation + test matrices
# ----------------------------
val_ok <- TRUE
val <- NULL
tryCatch({
  val <- build_matrix(MODELS, "val")
}, error = function(e) {
  val_ok <<- FALSE
  message("Validation files could not be used: ", conditionMessage(e))
})

test <- build_matrix(MODELS, "test")

use_test_for_fit <- FALSE
if (!val_ok) {
  if (!ALLOW_TEST_FIT_IF_NO_VAL) {
    stop("Validation predictions unavailable or inconsistent.\n",
         "Fix pred_val files or set ALLOW_TEST_FIT_IF_NO_VAL=TRUE (NOT recommended).")
  }
  message("Falling back to fitting weights on TEST (NOT recommended).")
  use_test_for_fit <- TRUE
  val <- test
}

y_fit <- val$y
P_fit <- val$P
y_test <- test$y
P_test <- test$P

cat("\nAligned sample sizes:\n")
cat(" - Fit set:", nrow(val$df), "\n")
cat(" - Test set:", nrow(test$df), "\n")

# ----------------------------
# Grey Wolf Optimizer in unconstrained space z; weights w=softmax(z)
# ----------------------------
D <- length(MODELS)
wolves <- matrix(rnorm(N_WOLVES * D, sd = 1), nrow = N_WOLVES, ncol = D)

eval_wolf <- function(z) {
  w <- softmax(z)
  p <- as.numeric(P_fit %*% w)
  metric(y_fit, p)
}

fitness <- apply(wolves, 1, eval_wolf)

ord <- order(fitness)
alpha <- wolves[ord[1], , drop = FALSE]
beta  <- wolves[ord[2], , drop = FALSE]
delta <- wolves[ord[3], , drop = FALSE]
alpha_fit <- fitness[ord[1]]

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

  fitness <- apply(wolves, 1, eval_wolf)
  ord <- order(fitness)

  alpha <- wolves[ord[1], , drop = FALSE]
  beta  <- wolves[ord[2], , drop = FALSE]
  delta <- wolves[ord[3], , drop = FALSE]
  alpha_fit <- fitness[ord[1]]

  if (t %% 25 == 0) {
    message(sprintf("Iter %d/%d | Best %s (fit set): %.8f", t, N_ITERS, OBJECTIVE, alpha_fit))
  }
}

best_w <- softmax(alpha[1, ])
names(best_w) <- MODELS

# ----------------------------
# Evaluate on TEST
# ----------------------------
p_ens_test <- as.numeric(P_test %*% best_w)

mae_test  <- mae(y_test, p_ens_test)
mse_test  <- mean((p_ens_test - y_test)^2)
rmse_test <- sqrt(mse_test)

ensemble_metrics <- data.frame(
  objective_fit_set = OBJECTIVE,
  used_test_for_fit = use_test_for_fit,
  n_models = length(MODELS),
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

pred_df <- test$df |> transmute(
  target_date = target_date,
  y_true = y_true,
  y_pred_ensemble = p_ens_test
)
write.csv(pred_df, preds_path, row.names = FALSE)

cat("\nSaved files:\n",
    " - ", weights_path, "\n",
    " - ", metrics_path, "\n",
    " - ", preds_path, "\n", sep = "")

cat("\nBest weights (sum should be 1):\n")
print(round(best_w, 6))
cat("Sum weights:", sum(best_w), "\n")
