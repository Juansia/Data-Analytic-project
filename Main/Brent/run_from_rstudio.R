# run_from_rstudio.R
# Option 3: Use RStudio as your IDE and run the existing Python workflow via RStudio's Terminal.
# Save this file in the SAME folder as main.py and requirements.txt, then run it in RStudio.
#
# What it does:
#  - Creates a local virtual environment (.venv) if missing
#  - Installs pip dependencies from requirements.txt
#  - Checks that dataset/processed_data_best_corr_sentiment.csv exists
#  - Runs: python main.py <MODEL_NAME>
#  - Optional: runs ensemble and/or Flask app
#
# Notes:
#  - This is NOT converting your project to R. It is an R "launcher" for your Python repo.
#  - If your main.py is NOT in this folder (e.g., it's under "Main/"), set SUBDIR <- "Main".

# ---------- 0) Must be in RStudio ----------
if (!requireNamespace("rstudioapi", quietly = TRUE)) install.packages("rstudioapi")
if (!rstudioapi::isAvailable()) {
  stop("This script must be run inside RStudio (it uses rstudioapi::terminalExecute).")
}

# ---------- 1) USER SETTINGS (edit these) ----------
MODEL_NAME   <- "Bi-GRU"   # e.g. "Bi-GRU", "CNN-BiLSTM-att", "torch-CNN-LSTM"
RUN_ENSEMBLE <- FALSE      # TRUE to run ensemble after the single model
RUN_APP      <- FALSE      # TRUE to run Flask app.py (keeps running in Terminal)

# If your runnable code lives in a subfolder (common example: 'Main'),
# put that folder name here. Leave "" if main.py is in the project root.
SUBDIR <- ""               # e.g. "Main"

# Expected dataset path (relative to project root OR SUBDIR folder)
DATASET_REL <- file.path("dataset", "processed_data_best_corr_sentiment.csv")

# Local virtual env directory (project-local)
VENV_DIR <- ".venv"

# ---------- 2) Resolve project + working directory ----------
proj_root <- tryCatch(rstudioapi::getActiveProject(), error = function(e) getwd())
if (is.null(proj_root) || proj_root == "") proj_root <- getwd()

work_dir <- if (nzchar(SUBDIR)) file.path(proj_root, SUBDIR) else proj_root
work_dir <- normalizePath(work_dir, winslash = "/", mustWork = FALSE)

# ---------- 3) Helpers ----------
term_run <- function(cmd) {
  message(">>> ", cmd)
  rstudioapi::terminalExecute(cmd, workingDir = work_dir, show = TRUE)
}

python_in_venv <- function() {
  if (.Platform$OS.type == "windows") {
    file.path(work_dir, VENV_DIR, "Scripts", "python.exe")
  } else {
    file.path(work_dir, VENV_DIR, "bin", "python")
  }
}

# ---------- 4) Print status ----------
message("Working directory: ", work_dir)
message("Expected dataset : ", file.path(work_dir, DATASET_REL))
message("Reminder: choose features in config/args.py (e.g., ['SENT','BRENT Close'] or ['SENT','USDX','BRENT Close']).")

# ---------- 5) Create venv if missing ----------
venv_python <- python_in_venv()
if (!file.exists(venv_python)) {
  term_run(sprintf('python -m venv "%s"', file.path(work_dir, VENV_DIR)))
} else {
  message("Venv already exists: ", venv_python)
}

# ---------- 6) Install dependencies ----------
# Use python -m pip to bind pip to the venv interpreter
term_run(sprintf('"%s" -m pip install --upgrade pip', venv_python))
term_run(sprintf('"%s" -m pip install -r "%s"', venv_python, file.path(work_dir, "requirements.txt")))

# ---------- 7) Dataset check ----------
if (!file.exists(file.path(work_dir, DATASET_REL))) {
  warning("Dataset not found. Put the CSV at: ", file.path(work_dir, DATASET_REL))
  warning("CSV must include at least columns: 'BRENT Close' and 'SENT' (USDX optional).")
} else {
  message("Dataset found ✅")
}

# ---------- 8) Run single model training + tuning ----------
# Equivalent to: python main.py [MODEL_NAME]
term_run(sprintf('"%s" "%s" "%s"', venv_python, file.path(work_dir, "main.py"), MODEL_NAME))

# ---------- 9) Optional: run ensemble ----------
if (RUN_ENSEMBLE) {
  # Your README says both forms are supported:
  # term_run(sprintf('"%s" "%s" --run_ensemble', venv_python, file.path(work_dir, "main.py")))
  term_run(sprintf('"%s" "%s" ensemble', venv_python, file.path(work_dir, "main.py")))
}

# ---------- 10) Optional: run Flask app ----------
if (RUN_APP) {
  term_run(sprintf('"%s" "%s"', venv_python, file.path(work_dir, "app.py")))
  message("Flask started. Open http://127.0.0.1:5000/ in your browser.")
  message("To test /predict in another Terminal, you can run:")
  message('curl -X POST http://127.0.0.1:5000/predict -d "days=5" -d "commodity=brent"')
}

message("Commands were sent to the RStudio Terminal. Check ./results and ./checkpoints after runs finish.")
