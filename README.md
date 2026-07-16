# Enhancing Subseasonal Predictability of Indian Monsoon Rainfall

Welcome to the code repository for our subseasonal-to-seasonal (S2S) monsoon downscaling project. This repository contains the complete deep learning and statistical post-processing pipeline used to correct NCMRWF Unified Model (NCUM) extended-range precipitation forecasts for the Indian Summer Monsoon. 

The core of this approach is a two-stage pipeline designed to solve the "accuracy vs. realism" trade-off inherent in deep learning precipitation models:
1. **Stage 1 (Spatiotemporal Correction):** A ConvLSTM network with Channel and Spatial Attention (CBAM) that learns the large-scale atmospheric envelope to correct the placement and timing of rainfall.
2. **Stage 2 (Variance Restoration):** A Quantile Delta Mapping (QDM) step applied directly to the network output to restore the realistic spatial variance and intensity distribution that neural networks typically smooth away.

---

## 🛠️ Prerequisites & Setup

### Data Requirements
To run these scripts out-of-the-box, you will need the following datasets pre-processed into weekly accumulations:
*   **Predictors:** NCUM extended-range hindcasts (e.g., `Week_1_AllYears.nc`).
*   **Target:** IMD gridded observation rainfall at 0.25° resolution.
*   **Ancillary:** An India state boundary shapefile (`.shp`) and a static orography/geopotential NetCDF file.
*   **Thresholds:** A unified climatological threshold file (containing p33, p50, p66 maps) for categorical metrics.

> **Important:** Before running any script, open it and update the file paths at the top (look for variables like `PREDICTOR_DIR`, `TARGET_DIR`, `INDIA_SHP_PATH`, etc.) to match your local or HPC directory structure.

### Python Environment
The code relies heavily on the standard scientific Python stack and deep learning libraries. Key dependencies include:
*   `tensorflow` (tested with GPUs)
*   `keras_tuner`
*   `xgboost`, `scikit-learn`
*   `xarray`, `netCDF4`, `pandas`, `numpy`, `scipy`
*   `geopandas`, `regionmask`, `cartopy`, `matplotlib`

---

## 🚀 Execution Order & Pipeline Guide

The pipeline is entirely modular. You should execute the scripts in the following sequential order:

### Step 1: Dynamic Feature Selection
**Script:** `Xgb_feature_importance_new_2.py`
Instead of hardcoding the predictor variables, this script dynamically selects them per lead week. 
*   It runs a robust XGBoost permutation importance loop with a moving-block bootstrap (`N_SEEDS = 10`) to ensure the transition from moisture-driven predictors (Week 1) to upper-level dynamics (Weeks 3-4) is physically robust and not a random artifact.
*   It filters for statistically significant predictors, enforces the inclusion of static downscaling anchors (orography and raw model precipitation), and checks for collinearity/redundancy.
*   It automatically saves the selected feature lists to a target directory, which the main deep learning script will read from.

### Step 2: Main Deep Learning & QDM Pipeline
**Script:** `Precipitation_correction_sub_seasonal_imd_qdm_no_blending_2.py`
This is the heavy lifter. It handles hyperparameter tuning via Hyperband, embargo-blocked Leave-Three-Year-Out (L3YO) cross-validation, and the two-stage M5 DL + QDM correction.
*   **To run a specific week:** `python Precipitation_correction_sub_seasonal_imd_qdm_no_blending_2.py --week 1`
*   **To measure the training noise floor (replicates):** Pass a custom seed like `--seed 99`. This skips tuning, reuses the production hyperparameters, and trains a new replicate to measure internal hardware variance.
*   It outputs the final `GoldStandard_L3YO_Results_Week{w}.nc` files containing the raw model, the EQM baseline, the DL-only output, and the final DL+QDM output.

### Step 3: Architectural Ablation Study
**Script:** `Abalation_study.py`
This script systematically disables components of the M5 architecture (CBAM attention blocks, residual connections, ConvLSTM bottleneck) to quantify exactly what each piece contributes. 

> **Parallel Execution Note:** While this script imports its deep learning layers directly from the Step 2 script to guarantee architectural consistency, its training loops are computationally independent. If you have access to multiple GPU nodes, **Step 2 and Step 3 can be executed simultaneously in parallel.**

*   **To train a week:** `python Abalation_study.py --week 1`
*   **To plot the results:** Run `python Abalation_study.py --combine` once all four weeks are finished. It calculates paired seasonal block-bootstrap statistics and outputs the temporal decay figures.

### Step 4: Evaluation & Plotting
Once your `GoldStandard` NetCDF files are generated, use these standalone scripts to generate publication-quality figures. 

*   **Continuous Metrics:** `Validation_plots_all_weeks_final_2.py`
    Calculates continuous and spatial skill scores (NMAE, Spearman Correlation, Wasserstein Distance, FSS). Generates both the univariate spatial maps and the bivariate skill decomposition (Correlation vs NMAE).
*   **Categorical Metrics:** `Unified_threshold_based_skill_scores.py`
    Calculates hit-based metrics (POD, FAR, ETS, HSS) against specific climatological thresholds (p33, p50, p66). Generates the dense "Mega Performance Portrait" heatmap showing statistical significance for every region, week, model, and threshold.
*   **Spatial Variance Check:** `spatial_spectra_variance_check.py`
    Extracts and plots the spatial power spectra across all wavenumbers to visually prove that the QDM stage successfully restores the high-frequency grid-scale variance that the neural network smoothed out.

---

## 🧠 Methodology Deep Dive: How the Model Works

Because precipitation at subseasonal leads suffers from severe spatial displacement and intensity errors, standard pixel-by-pixel downscaling fails. This pipeline implements a sequence-to-image architecture with custom loss functions to solve this.

### The M5 Spatiotemporal Architecture
The `build_M5_model` function defines an Encoder-Bottleneck-Decoder structure:
* **Spatial Encoder:** Uses `TimeDistributed` 2D Convolutions and Residual Blocks (`ResBlock`) to extract spatial features independently from each of the 6 weeks in the input sequence. A gap guard ensures sequences do not bridge the 9-month winter gap.
* **Temporal Bottleneck:** A `ConvLSTM2D` layer processes the sequence of spatial feature maps to learn the temporal evolution (e.g., the propagation of the monsoon intraseasonal oscillation).
* **Attention Mechanism:** A Convolutional Block Attention Module (`CBAMBlock`) combines Channel Attention (which meteorological variables matter most right now?) and Spatial Attention (where are the critical synoptic features located?).
* **Skip-Connected Decoder:** Upsamples the temporal bottleneck's output, concatenating it with high-resolution spatial maps bypassed from the encoder (`MatchShapes`) to rebuild the local grid-scale detail.

### The Custom Composite Loss
Training a neural network on precipitation using standard Mean Squared Error (MSE) results in highly smoothed, unrealistic drizzle fields. To combat this, the model is compiled with a custom `CombinedLoss`:
* **Tweedie Loss:** Specifically handles the heavy-tailed, zero-inflated nature of weekly rainfall.
* **Neighborhood MAE:** Instead of penalizing the model for putting rain in the adjacent grid cell, the prediction and ground truth are average-pooled over a 3x3 window before the absolute error is calculated (`neighborhood_mae`). This relaxes the spatial grid-locking requirement, which is physically unpredictable 3-4 weeks out.
* **Structural Similarity Index:** The `ssim_loss` forces the model to replicate the "clumpy", highly organized spatial structure of the observed IMD rainfall patterns. An ocean mask is dynamically applied so dry marine pixels do not artificially inflate the score.

### Variance Restoration
Even with SSIM, the deep learning output suppresses high-frequency grid-scale variance. The pipeline fixes this via Quantile Delta Mapping (`apply_qdm_pooled`). Because the M5 neural network successfully corrects the temporal *timing* of wet and dry weeks, the script applies QDM directly to the network's output. QDM is a monotone, rank-preserving transform that restores extreme rainfall intensities without destroying the temporal rank correlation that the deep learning stage just fixed.

### Rigorous Cross-Validation
To prevent data leakage caused by the high temporal autocorrelation of the monsoon, the script utilizes an `embargo_blocked_fold` generator. Testing is strictly blocked by 3-year chunks (Leave-Three-Year-Out). A 1-year buffer is aggressively dropped from the training set on either side of the test block and the validation year, guaranteeing the model never trains on a meteorological state adjacent to what it will be tested on.

---

## 🖥️ Hardware Notes
*   **GPUs:** The deep learning scripts (`Precipitation_correction...` and `Abalation_study.py`) are configured to isolate GPUs based on the lead week being processed (e.g., Week 1 uses GPU 0, Week 2 uses GPU 1). You can run all four weeks simultaneously if you have a 4-GPU node.
*   **CPUs:** The plotting scripts aggressively flatten bootstrap loops into micro-tasks to max out HPC CPU cores (up to 96+ parallel threads). They will execute in minutes on a heavy compute node but will freeze a standard laptop. Adjust the `n_workers` variable in the `run()` block of the plotting scripts if you need to limit thread usage on a personal machine.
