# ============================================================
# ARIMA / ARIMAX (with optional exogenous regressors) in R
# Uses: dataset/processed_data_best_corr_sentiment.csv
#
# Output:
#   - arima_predictions_h3.csv
#   - metrics printed to console
#
# Notes:
#   - Horizon: 3 trading days ahead (h = 3)
#   - Split: 80% train / 10% validation / 10% test (chronological)
#   - Evaluation: rolling-origin (walk-forward), expanding window
#   - ARIMAX uses xreg if USDX and/or sentiment columns are detected
# ============================================================

# ---- Packages ----
pkgs <- c("readr", "dplyr", "tidyr", "lubridate", "forecast")
to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

# ---- User settings ----
file_path <- file.path("dataset", "processed_data_best_corr_sentiment.csv")
h <- 3  # 3-day ahead horizon

# If auto-detection fails, set these manually (uncomment & edit):
# date_col   <- "Date"
# target_col <- "BZ_Close"
# xreg_cols  <- c("USDX", "SENT")

# ---- Load data ----
if (!file.exists(file_path)) {
  stop(paste0("CSV not found at: ", file_path,
              "\nPlace processed_data_best_corr_sentiment.csv inside a folder named 'dataset' ",
              "in your project working directory."))
}
df <- readr::read_csv(file_path, show_col_types = FALSE)

# ---- Helper: choose a column by name pattern ----
pick_col <- function(cols, patterns) {
  for (p in patterns) {
    hit <- cols[grepl(p, cols, ignore.case = TRUE)]
    if (length(hit) > 0) return(hit[1])
  }
  return(NA_character_)
}

cols <- names(df)

# ---- Auto-detect columns (only if not manually set above) ----
if (!exists("date_col")) {
  date_col <- pick_col(cols, c("^date$", "date", "time", "timestamp"))
}
if (is.na(date_col)) {
  message("Could not auto-detect a date column. Columns are:\n  ", paste(cols, collapse = ", "))
  stop("Please set date_col manually near the top of this script.")
}

if (!exists("target_col")) {
  # Prefer brent/bz first; fall back to close/price if needed
  brent_like <- cols[grepl("brent|bz", cols, ignore.case = TRUE)]
  if (length(brent_like) > 0) {
    target_col <- brent_like[1]
  } else {
    target_col <- pick_col(cols, c("close", "price"))
  }
}
if (is.na(target_col)) {
  message("Could not auto-detect a target column. Columns are:\n  ", paste(cols, collapse = ", "))
  stop("Please set target_col manually near the top of this script.")
}

if (!exists("xreg_cols")) {
  usdx_col <- pick_col(cols, c("usdx", "dxy", "dollar", "dx"))
  sent_col <- pick_col(cols, c("sent", "crudebert", "finbert", "csum", "headline"))
  xreg_cols <- c()
  if (!is.na(usdx_col)) xreg_cols <- c(xreg_cols, usdx_col)
  if (!is.na(sent_col)) xreg_cols <- c(xreg_cols, sent_col)
}

message("Detected columns:")
message("  date_col   = ", date_col)
message("  target_col = ", target_col)
message("  xreg_cols  = ", ifelse(length(xreg_cols) == 0, "(none)", paste(xreg_cols, collapse = ", ")))

# ---- Clean / sort ----
df <- df %>%
  mutate(
    .date = suppressWarnings(lubridate::ymd(.data[[date_col]]))
  )

# If ymd failed, try base parsing
if (all(is.na(df$.date))) {
  df <- df %>% mutate(.date = as.Date(.data[[date_col]]))
}
if (all(is.na(df$.date))) {
  stop("Date parsing failed. Convert your date column to YYYY-MM-DD or set a custom parse.")
}

df <- df %>%
  arrange(.date) %>%
  filter(!is.na(.data[[target_col]]))

keep_cols <- c(".date", target_col, xreg_cols)
df <- df %>% select(all_of(keep_cols))

# Drop NA rows in regressors (only if using ARIMAX)
if (length(xreg_cols) > 0) {
  df <- df %>% tidyr::drop_na(all_of(xreg_cols))
}

n <- nrow(df)
if (n < 60) warning("Dataset is small; ARIMA evaluation may be unstable (n < 60).")

# ---- Chronological split (80/10/10) ----
train_end <- floor(0.8 * n)
val_end   <- floor(0.9 * n)

# ---- Rolling-origin forecast function ----
# For each origin i, fit on rows 1:i then forecast h steps.
# We store the h-step forecast as prediction for row (i+h).
rolling_forecast <- function(df, y_col, xreg_cols = NULL, start_i, end_i, h = 3) {
  y <- df[[y_col]]
  preds <- rep(NA_real_, nrow(df))

  for (i in start_i:end_i) {
    y_train <- y[1:i]

    if (is.null(xreg_cols) || length(xreg_cols) == 0) {
      fit <- forecast::auto.arima(y_train)
      fc  <- forecast::forecast(fit, h = h)
    } else {
      x_train <- as.matrix(df[1:i, xreg_cols, drop = FALSE])

      # Fit ARIMAX
      fit <- forecast::auto.arima(y_train, xreg = x_train)

      # Need future xreg for i+1 ... i+h
      if (i + h > nrow(df)) break
      x_future <- as.matrix(df[(i + 1):(i + h), xreg_cols, drop = FALSE])

      fc <- forecast::forecast(fit, h = h, xreg = x_future)
    }

    if (i + h <= length(preds)) {
      preds[i + h] <- as.numeric(fc$mean[h])
    }
  }
  preds
}

# Origins for validation + test (expanding window):
# - Start fitting at end of training set
# - End at n - h so i+h is valid
origin_start <- train_end
origin_end   <- n - h

pred_arima <- rolling_forecast(df, target_col, NULL, origin_start, origin_end, h)

pred_arimax <- NULL
if (length(xreg_cols) > 0) {
  pred_arimax <- rolling_forecast(df, target_col, xreg_cols, origin_start, origin_end, h)
}

# ---- Metrics ----
rmse <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
mae  <- function(a, p) mean(abs(a - p), na.rm = TRUE)
mse  <- function(a, p) mean((a - p)^2, na.rm = TRUE)

score_period <- function(actual, pred, idx_start, idx_end) {
  a <- actual[idx_start:idx_end]
  p <- pred[idx_start:idx_end]
  data.frame(
    MAE  = mae(a, p),
    MSE  = mse(a, p),
    RMSE = rmse(a, p)
  )
}

actual <- df[[target_col]]

# Prediction indices that exist begin at train_end + h
val_pred_start  <- train_end + h
val_pred_end    <- val_end
test_pred_start <- val_end + h
test_pred_end   <- n

cat("\n============================\n")
cat("ARIMA (price-only)\n")
cat("============================\n")
cat("Validation metrics:\n")
print(score_period(actual, pred_arima, val_pred_start, val_pred_end))
cat("Test metrics:\n")
print(score_period(actual, pred_arima, test_pred_start, test_pred_end))

if (!is.null(pred_arimax)) {
  cat("\n============================\n")
  cat("ARIMAX (with xreg)\n")
  cat("============================\n")
  cat("xreg cols: ", paste(xreg_cols, collapse = ", "), "\n", sep = "")
  cat("Validation metrics:\n")
  print(score_period(actual, pred_arimax, val_pred_start, val_pred_end))
  cat("Test metrics:\n")
  print(score_period(actual, pred_arimax, test_pred_start, test_pred_end))
} else {
  cat("\nARIMAX not run because no xreg columns were detected.\n")
}

# ---- Save predictions ----
out <- df %>%
  mutate(
    pred_arima  = pred_arima,
    pred_arimax = if (!is.null(pred_arimax)) pred_arimax else NA_real_
  )

readr::write_csv(out, "arima_predictions_h3.csv")
cat("\nSaved: arima_predictions_h3.csv\n")
