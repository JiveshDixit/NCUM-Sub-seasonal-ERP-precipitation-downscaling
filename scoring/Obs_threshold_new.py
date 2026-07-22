"""
Threshold Sensitivity Diagnostic (L3YO-consistent)
--------------------------------------------------
Quantifies how much the categorical precipitation thresholds (p33, p50, p66, p90) 
drift when computed on the TRAINING YEARS of each Leave-Three-Year-Out (L3YO) fold,
instead of the FULL climatological record.

Why this matters:
Categorical thresholds are derived solely from observations, so using the full record 
is not train/test leakage (the model has already made its prediction). However, it is 
a verification-reference choice. This script measures if that choice artificially 
changes the scoring.

Decision Rule:
- Class-flip fraction < 0.02 (2%): Full-record thresholds are immaterial. 
  Safe to use the unified threshold map for evaluation consistency.
- Class-flip fraction > 0.05 (5%): Material drift. Per-fold thresholds are required.
"""

import os
import numpy as np
import xarray as xr
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration ---
IMD_DIR     = "/home/ncmrwf/bcwc/jivesh/hindcast_erp/Obs_precip"
IMD_PATTERN = "IMD_week*_sum_25km_on_model_t.nc"
VAR_NAME    = "tp_weekly_sum"
PERCENTILES = [0.33, 0.50, 0.66, 0.90]

LTYO_BLOCK  = 3

def main():
    print("Loading IMD JJAS weekly files...")
    
    # Load and concatenate IMD observations
    ds = xr.open_mfdataset(os.path.join(IMD_DIR, IMD_PATTERN), combine="nested", concat_dim="t", parallel=False)
    ds = ds.load()
    
    jjas = ds["t"].dt.month.isin([6, 7, 8, 9])
    ds = ds.sel(t=jjas).drop_duplicates(dim="t")
    precip = ds[VAR_NAME]

    years = ds["t"].dt.year.values
    uniq_years = np.unique(years)
    print(f"  {len(uniq_years)} seasons ({uniq_years.min()}-{uniq_years.max()}), {precip.sizes['t']} weekly samples")

    # --- Construct L3YO Folds ---
    blocks = [uniq_years[i:i + LTYO_BLOCK] for i in range(0, len(uniq_years), LTYO_BLOCK)]
    print(f"  {len(blocks)} L3YO folds: " + ", ".join("[" + "-".join(str(int(y)) for y in b) + "]" for b in blocks))

    # --- Compute Full-Record Thresholds (The Reference) ---
    print("\nComputing FULL-record thresholds...")
    thr_full = precip.quantile(PERCENTILES, dim="t", skipna=True)

    precip_np = precip.values  # Extract to numpy once for speed (t, lat, lon)
    finite_samples = np.isfinite(precip_np)
    n_finite = finite_samples.sum()

    absdiff_mean = {p: [] for p in PERCENTILES}
    absdiff_p95  = {p: [] for p in PERCENTILES}
    flip_frac    = {p: [] for p in PERCENTILES}

    fold_threshold_list = []
    
    # --- Evaluate Train-Only Folds ---
    print(f"Evaluating {len(blocks)} train-only folds...\n")
    for bi, test_block in enumerate(blocks):
        train_mask_years = ~np.isin(uniq_years, test_block)
        keep_years = uniq_years[train_mask_years]
        t_mask = np.isin(years, keep_years)

        # Calculate threshold using only the training years
        thr_sub = precip.isel(t=t_mask).quantile(PERCENTILES, dim="t", skipna=True)
        thr_sub_expanded = thr_sub.assign_coords(fold=bi+1).expand_dims("fold")
        fold_threshold_list.append(thr_sub_expanded)

        for p in PERCENTILES:
            a = thr_full.sel(quantile=p).values   # full-record threshold field
            b = thr_sub.sel(quantile=p).values    # train-only threshold field

            # 1 & 2: Threshold drift, in percent, over valid cells
            denom = np.where(np.abs(a) > 1e-6, np.abs(a), np.nan)
            pct = np.abs(b - a) / denom * 100.0
            pct = pct[np.isfinite(pct)]
            
            absdiff_mean[p].append(np.nanmean(pct))
            absdiff_p95[p].append(np.nanpercentile(pct, 95))

            # 3. Class-flip fraction over ALL (cell, week) samples
            cls_full = precip_np >= a[None, :, :]
            cls_sub  = precip_np >= b[None, :, :]
            flips = (cls_full != cls_sub) & finite_samples
            flip_frac[p].append(flips.sum() / n_finite)

        print(f"  Fold {bi+1}/{len(blocks)}  held out [{'-'.join(str(int(y)) for y in test_block)}]  "
              + "  ".join(f"p{int(p*100)} flip={flip_frac[p][-1]*100:5.2f}%" for p in PERCENTILES))

    # --- Summary & Interpretation ---
    print("\n" + "=" * 74)
    print("  THRESHOLD SENSITIVITY  (per-fold train-only vs full record)")
    print("=" * 74)
    print(f"{'Percentile':>10} | {'Mean |%diff|':>13} | {'95th |%diff|':>13} | {'CLASS-FLIP frac':>16}")
    print("-" * 74)
    
    worst_flip = 0.0
    for p in PERCENTILES:
        mf = np.mean(flip_frac[p])
        worst_flip = max(worst_flip, mf)
        print(f"{int(p*100):>9}% | {np.mean(absdiff_mean[p]):>12.2f}% | "
              f"{np.mean(absdiff_p95[p]):>12.2f}% | {mf*100:>15.2f}%")
    print("-" * 74)

    print("\nInterpretation:")
    print("  The CLASS-FLIP fraction is the quantity that actually changes the")
    print("  categorical scores (POD/FAR/ETS). Threshold %-drift alone is not sufficient,")
    print("  because a shift at a steep part of the CDF flips many more samples than")
    print("  the same shift at a flat part.\n")
    
    if worst_flip < 0.02:
        print(f"  -> Worst-case mean class-flip is {worst_flip*100:.2f}% (< 2%).")
        print("     Full-record thresholds are immaterial. Reporting this number is")
        print("     sufficient. Preferred: still use per-fold train-only thresholds")
        print("     and cite this as confirmation that the choice does not matter.")
    elif worst_flip < 0.05:
        print(f"  -> Worst-case mean class-flip is {worst_flip*100:.2f}% (2-5%).")
        print("     Borderline. Use per-fold train-only thresholds to be safe.")
    else:
        print(f"  -> Worst-case mean class-flip is {worst_flip*100:.2f}% (> 5%).")
        print("     Material. Per-fold train-only thresholds are required.")
    print()

    # --- Save Output ---
    print("\nSaving per-fold train-only thresholds to a single NetCDF file...")
    thresholds_ds = xr.concat(fold_threshold_list, dim="fold").to_dataset(name="thresholds")
    save_name = "IMD_JJAS_TrainOnly_Thresholds_L3YO.nc"
    thresholds_ds.to_netcdf(save_name)

if __name__ == "__main__":
    main()