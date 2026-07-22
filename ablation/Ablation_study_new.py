"""
Architectural Ablation Study (M0 to M5)
---------------------------------------
Breaks down the M5 deep learning architecture into smaller structural variants 
to isolate the predictive contribution of each neural component (ConvLSTM, CBAM, Skips).

Design Approach:
1. Evaluates identical folds and hyperparameters to test pure architecture, not capacity.
2. Evaluates spatial (FSS) and threshold (POD) metrics alongside pixel error (MAE/RMSE)
   to explicitly demonstrate the double-penalty problem of precipitation verification.
3. Utilizes 3D vectorized metrics and parallel block bootstrapping for rapid evaluation.

Usage:
  for w in 1 2 3 4; do python Ablation_study.py --week $w & done; wait
  python Ablation_study.py --combine
"""

import os
import argparse
import warnings
import logging
import gc

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

# 1. Argparse Setup
parser = argparse.ArgumentParser()
parser.add_argument('--week', type=int, choices=[1, 2, 3, 4], help='Lead week (1-4)')
parser.add_argument('--combine', action='store_true', help='Merge per-week CSVs into final dashboard.')
args, _unknown = parser.parse_known_args()

# 2. Scientific Libraries & GPU Binding
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
import geopandas as gpd
import regionmask
import tensorflow as tf

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# --- Optimization Settings ---
SEED = 42
USE_MIXED_PRECISION = False   
USE_XLA = False               
N_JOBS = min(os.cpu_count() or 4, 32)

if not args.combine and args.week is not None:
    gpus = tf.config.list_physical_devices('GPU')
    gpu_idx = args.week - 1
    if gpus and gpu_idx < len(gpus):
        try:
            tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
            print(f"--- Ablation Week {args.week} bound to GPU {gpu_idx}: {gpus[gpu_idx].name} ---")
        except RuntimeError as e:
            print(e)

tf.config.threading.set_intra_op_parallelism_threads(24)
tf.config.threading.set_inter_op_parallelism_threads(24)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Import Custom Pipeline Layers
import Precipitation_correction_sub_seasonal_imd_qdm_no_blending as tp
from Precipitation_correction_sub_seasonal_imd_qdm_no_blending import (
    CombinedLoss, CBAMBlock, ResBlock, TakeLastStep, MatchShapes,
    neighborhood_mae, tweedie_metric_monitor, ssim_metric_monitor,
)
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled.")

# --- Configuration Paths ---
PREDICTOR_DIR  = "/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/erfgc2/final_weekly_timeseries"
TARGET_DIR     = os.path.expanduser("~/hindcast_erp/Obs_precip")
INDIA_SHP_PATH = "/home/ncmrwf/bcwc/jivesh/Shape_files_India/India_State_Boundary_Updated.shp"
THRESHOLD_FILE = "/home/ncmrwf/bcwc/jivesh/hindcast_erp/DESN/IMD_JJAS_Unified_Thresholds_25km.nc"

tp.PREDICTOR_DIR = PREDICTOR_DIR
tp.TARGET_DIR = TARGET_DIR
tp.XGB_RESULTS_DIR = "./../new/FEATURE_IMPORTANCE_RESULTS_ROBUST_REDUNDANCY_False"

FSS_PERCENTILE = "p66"
SEQUENCE_LENGTH = 6
BATCH_SIZE = 8
PREDICT_BATCH = 32

# Shared Hyperparameters
LR = 3e-4
DROPOUT = 0.1
LSTM_UNITS = 256
BASE_FILTER = 128
ABL_EPOCHS = 120
ABL_PATIENCE = 15

MODELS = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M5_noCBAM']
LTYO_BLOCK = 3
PLOT_METRIC = 'fss'

CONTRAST_PAIRS = [('M3', 'M4'), ('M5_noCBAM', 'M5')]
BOOT_N = 1000
BOOT_ALPHA = 0.10     

REGION_BOXES = {
    'Hilly':       (28.0, 37.0, 72.0, 97.0),
    'Northwest':   (24.0, 30.0, 68.0, 78.0),
    'CentralNE':   (21.0, 27.0, 78.0, 88.0),
    'Northeast':   (22.0, 29.0, 89.0, 97.0),
    'WestCentral': (16.0, 24.0, 72.0, 80.0),
    'Peninsular':  (8.0,  16.0, 74.0, 80.0),
    'AllIndia':    (6.0,  38.0, 66.0, 100.0),
}

def out_csv(week):       return f"ablation_metrics_LTYO_W{week}.csv"
def out_contrast(week):  return f"ablation_contrasts_LTYO_W{week}.csv"

# --- Variant Builder ---
def build_variant(input_shape, model_name, loss_obj, lr=LR, dropout=DROPOUT, lstm_units=LSTM_UNITS, base_filter=BASE_FILTER):
    flags = {
        'M0':        dict(use_skips=False, use_lstm=False, use_cbam=False, use_residual=False),
        'M1':        dict(use_skips=True,  use_lstm=False, use_cbam=False, use_residual=False),
        'M2':        dict(use_skips=True,  use_lstm=True,  use_cbam=False, use_residual=False),
        'M3':        dict(use_skips=True,  use_lstm=False, use_cbam=True,  use_residual=False),
        'M4':        dict(use_skips=True,  use_lstm=True,  use_cbam=True,  use_residual=False),
        'M5':        dict(use_skips=True,  use_lstm=True,  use_cbam=True,  use_residual=True),
        'M5_noCBAM': dict(use_skips=True,  use_lstm=True,  use_cbam=False, use_residual=True),
    }[model_name]
    
    use_skips, use_lstm = flags['use_skips'], flags['use_lstm']
    use_cbam, use_residual = flags['use_cbam'], flags['use_residual']
    f1, f2, f3 = base_filter, base_filter * 2, base_filter * 4

    def block(x, filters, td=False):
        if use_residual:
            layer = ResBlock(filters, use_cbam=use_cbam)
            return layers.TimeDistributed(layer)(x) if td else layer(x)
        if td:
            x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding='same', activation='relu'))(x)
            x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding='same', activation='relu'))(x)
            if use_cbam: x = layers.TimeDistributed(CBAMBlock(filters))(x)
            return x
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        if use_cbam: x = CBAMBlock(filters)(x)
        return x

    inputs = layers.Input(shape=input_shape)
    ref_full = TakeLastStep()(inputs)

    # Encoder
    x = layers.TimeDistributed(layers.Conv2D(f1, 3, padding='same'))(inputs)
    x = block(x, f1, td=True); enc1 = x
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(layers.Conv2D(f2, 3, padding='same'))(x)
    x = block(x, f2, td=True); enc2 = x
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(layers.Conv2D(f3, 3, padding='same'))(x)
    x = block(x, f3, td=True)

    # Bottleneck
    if use_lstm:
        x = layers.ConvLSTM2D(lstm_units, 3, padding='same', return_sequences=False)(x)
    else:
        x = TakeLastStep()(x)
        x = layers.Conv2D(lstm_units, 3, padding='same', activation='relu')(x)
    x = layers.SpatialDropout2D(dropout)(x)
    if use_cbam: x = CBAMBlock(lstm_units)(x)

    # Decoder
    x = layers.UpSampling2D()(x)
    if use_skips:
        s2 = TakeLastStep()(enc2)
        x = layers.Concatenate()([MatchShapes()([x, s2]), s2])
    x = block(x, f2, td=False)
    x = layers.SpatialDropout2D(dropout)(x)

    x = layers.UpSampling2D()(x)
    if use_skips:
        s1 = TakeLastStep()(enc1)
        x = layers.Concatenate()([MatchShapes()([x, s1]), s1])
    x = block(x, f1, td=False)
    x = MatchShapes()([x, ref_full])

    outputs = layers.Conv2D(1, 1, activation='softplus', dtype='float32')(x)
    model = Model(inputs, outputs, name=model_name)

    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_obj, jit_compile=USE_XLA,
                  metrics=[neighborhood_mae, tweedie_metric_monitor, ssim_metric_monitor])
    return model

# --- Evaluation Metrics ---
def split_LTYO_embargo(years_per_sample, block=LTYO_BLOCK):
    unique_years = np.unique(years_per_sample)
    folds = []
    for start in range(0, len(unique_years), block):
        test_years = unique_years[start:start + block]
        tr_m, va_m, te_m, _, _ = tp.embargo_blocked_fold(years_per_sample, unique_years, test_years, embargo=1)
        tr, va, te = np.where(tr_m)[0], np.where(va_m)[0], np.where(te_m)[0]
        if len(te) > 0: folds.append((tr, va, te))
    return folds

def region_mask(lats, lons, box):
    la0, la1, lo0, lo1 = box
    return ((lats[:, None] >= la0) & (lats[:, None] <= la1) & (lons[None, :] >= lo0) & (lons[None, :] <= lo1))

def _err_on(pred, truth, valid): return (pred - truth)[:, valid]

def regional_rmse(pred, truth, rmask, land2d):
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    e = _err_on(pred, truth, valid)
    return float(np.sqrt(np.nanmean(e ** 2)))

def regional_mae(pred, truth, rmask, land2d):
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    e = _err_on(pred, truth, valid)
    return float(np.nanmean(np.abs(e)))

def regional_bias(pred, truth, rmask, land2d):
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    e = _err_on(pred, truth, valid)
    return float(np.nanmean(e))

def regional_fss(pred, truth, rmask, land2d, threshold_map, window=5):
    """Calculates FSS using 3D vectorization for maximum speed."""
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    
    p_bin = ((pred >= threshold_map) & (valid == 1.0)).astype(float)
    t_bin = ((truth >= threshold_map) & (valid == 1.0)).astype(float)
    v_mask = np.broadcast_to(valid, p_bin.shape).astype(float)
    
    # 3D uniform filter over spatial axes (1, window, window)
    p_f = uniform_filter(p_bin, size=(1, window, window), mode='constant', cval=0.0)
    t_f = uniform_filter(t_bin, size=(1, window, window), mode='constant', cval=0.0)
    v_f = uniform_filter(v_mask, size=(1, window, window), mode='constant', cval=0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        p_frac = np.where(v_f > 0, p_f / v_f, 0.0)
        t_frac = np.where(v_f > 0, t_f / v_f, 0.0)
    
    valid_3d = np.broadcast_to(valid, p_bin.shape)
    mse_sum = np.nansum(np.where(valid_3d, (p_frac - t_frac)**2, 0.0))
    denom_sum = np.nansum(np.where(valid_3d, p_frac**2 + t_frac**2, 0.0))
        
    return float(1.0 - (mse_sum / denom_sum)) if denom_sum > 0 else np.nan

def regional_pod(pred, truth, rmask, land2d, threshold_map):
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    hits = np.sum((pred >= threshold_map) & (truth >= threshold_map) & valid)
    misses = np.sum((pred < threshold_map) & (truth >= threshold_map) & valid)
    return float(hits / (hits + misses)) if (hits + misses) > 0 else np.nan

# --- Parallel Bootstrapping ---
def _metric_eval(pred_3d, truth_3d, rmask, land2d, kind, p66_map):
    if kind == 'fss':  return regional_fss(pred_3d, truth_3d, rmask, land2d, p66_map, window=5)
    if kind == 'pod':  return regional_pod(pred_3d, truth_3d, rmask, land2d, p66_map)
    if kind == 'rmse': return regional_rmse(pred_3d, truth_3d, rmask, land2d)
    if kind == 'mae':  return regional_mae(pred_3d, truth_3d, rmask, land2d)
    return regional_bias(pred_3d, truth_3d, rmask, land2d)

def _boot_worker(draw_seasons, by_season, variant_preds, truth, pairs, region_masks, land2d, p66_map, metrics):
    """Worker function for executing a single bootstrap draw."""
    sel = np.concatenate([by_season[s] for s in draw_seasons])
    t_sel = truth[sel]
    res = {}
    for (A, B) in pairs:
        pA, pB = variant_preds[A][sel], variant_preds[B][sel]
        for r in region_masks:
            for mk in metrics:
                res[(r, A, B, mk)] = _metric_eval(pB, t_sel, region_masks[r], land2d, mk, p66_map) - \
                                     _metric_eval(pA, t_sel, region_masks[r], land2d, mk, p66_map)
    return res

def compute_contrasts(variant_preds, truth, season_ids, region_masks, land2d, p66_map, 
                      pairs=CONTRAST_PAIRS, metrics=('fss', 'pod', 'mae'), n_boot=BOOT_N, alpha=BOOT_ALPHA, seed=SEED):
    seasons = np.unique(season_ids)
    by_season = {s: np.where(season_ids == s)[0] for s in seasons}
    rng = np.random.default_rng(seed)

    # Base Point Estimate
    acc = {}
    for (A, B) in pairs:
        for r in region_masks:
            for mk in metrics:
                pt = _metric_eval(variant_preds[B], truth, region_masks[r], land2d, mk, p66_map) - \
                     _metric_eval(variant_preds[A], truth, region_masks[r], land2d, mk, p66_map)
                acc[(r, A, B, mk)] = {'point': pt, 'boot': np.empty(n_boot, dtype=float)}

    # Parallel Bootstrap Resampling
    draws = [rng.choice(seasons, size=len(seasons), replace=True) for _ in range(n_boot)]
    boot_results = Parallel(n_jobs=N_JOBS)(
        delayed(_boot_worker)(d, by_season, variant_preds, truth, pairs, region_masks, land2d, p66_map, metrics) for d in draws
    )

    # Accumulate
    for b, res in enumerate(boot_results):
        for key, val in res.items():
            acc[key]['boot'][b] = val

    rows = []
    for (r, A, B, mk), v in acc.items():
        lo, hi = float(np.percentile(v['boot'], 100 * alpha / 2)), float(np.percentile(v['boot'], 100 * (1 - alpha / 2)))
        rows.append(dict(region=r, pair=f"{A}->{B}", metric=mk, delta=v['point'], ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0)))
    return rows

# --- Main Training & Evaluation Loop ---
def run_ablation(week):
    print(f"\n--- Running Lead Week {week} ---")
    p_fp = os.path.join(PREDICTOR_DIR, f"Week_{week}_AllYears.nc")
    t_fp = os.path.join(TARGET_DIR, f"IMD_week{week}_sum_25km_on_model_t.nc")

    X_da, Y_da = tp.load_and_preprocess(p_fp, t_fp, week)
    X_seq, Y_seq, _, Ytrue_da = tp.create_jjas_sequences(X_da, Y_da, Y_da, SEQUENCE_LENGTH)
    if len(X_seq) == 0:
        print("  No valid sequences found; aborting."); return

    years_filtered = pd.to_datetime(Ytrue_da.t.values).year.values
    lats, lons = X_da.latitude.values, X_da.longitude.values

    if os.path.exists(INDIA_SHP_PATH):
        gdf = gpd.read_file(INDIA_SHP_PATH)
        land2d = ~np.isnan(regionmask.Regions(gdf.geometry).mask(lons, lats).values)
    else:
        land2d = np.ones((len(lats), len(lons)), bool)
    region_masks = {name: region_mask(lats, lons, box) for name, box in REGION_BOXES.items()}

    folds = split_LTYO_embargo(years_filtered, LTYO_BLOCK)
    truth_concat  = np.concatenate([Y_seq[te].squeeze(-1) for (_, _, te) in folds], 0)
    season_concat = np.concatenate([years_filtered[te] for (_, _, te) in folds], 0)

    print(f"  Loading unified '{FSS_PERCENTILE}' threshold map from {THRESHOLD_FILE}...")
    with xr.open_dataset(THRESHOLD_FILE) as ds_thr:
        renames = {k: v for k, v in [('lat', 'latitude'), ('lon', 'longitude')] if k in ds_thr.coords}
        if renames: ds_thr = ds_thr.rename(renames)
        p66_map = ds_thr[FSS_PERCENTILE].sel(latitude=lats, longitude=lons, method="nearest").values

    shared_loss = CombinedLoss(w_tweedie=0.15, w_mae=1.0, w_ssim=15.0)
    rows = []
    variant_preds = {}            
    
    for model_name in MODELS:
        print(f"\n  --- Training {model_name} ---")
        preds_all = []
        pf_rmse, pf_mae, pf_ssim, pf_nmae = [], [], [], []
        tr_losses, va_losses, stop_eps = [], [], []

        for fi, (tr, va, te) in enumerate(folds):
            mu, sd = X_seq[tr].mean(axis=(0, 1, 2, 3)), X_seq[tr].std(axis=(0, 1, 2, 3)) + 1e-7
            X_trn, X_van, X_ten = [(np.nan_to_num((x - mu) / sd)) for x in (X_seq[tr], X_seq[va], X_seq[te])]

            tf.keras.backend.clear_session()
            model = build_variant(X_seq.shape[1:], model_name, shared_loss)
            hist = model.fit(
                X_trn, Y_seq[tr], validation_data=(X_van, Y_seq[va]),
                epochs=ABL_EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                callbacks=[EarlyStopping('val_loss', patience=ABL_PATIENCE, restore_best_weights=True),
                           ReduceLROnPlateau('val_loss', factor=0.5, patience=4)],
            )
            tr_losses.append(hist.history['loss'][-1])
            va_losses.append(hist.history.get('val_loss', [np.nan])[-1])
            stop_eps.append(len(hist.history['loss']))

            eval_metrics = model.evaluate(X_ten, Y_seq[te], batch_size=PREDICT_BATCH, verbose=0)
            metric_dict = dict(zip(model.metrics_names, eval_metrics))
            pf_ssim.append(metric_dict.get('ssim_metric_monitor', np.nan))
            pf_nmae.append(metric_dict.get('neighborhood_mae', np.nan))

            p = model.predict(X_ten, batch_size=PREDICT_BATCH, verbose=0).squeeze(-1)
            t = Y_seq[te].squeeze(-1)
            preds_all.append(p)
            pf_rmse.append(regional_rmse(p, t, region_masks['AllIndia'], land2d))
            pf_mae.append(regional_mae(p, t, region_masks['AllIndia'], land2d))
            print(f"    Fold {fi+1}/{len(folds)} stopped at epoch {stop_eps[-1]}, val_loss={va_losses[-1]:.4f}")
            del model; gc.collect()

        if not preds_all: continue
        preds = np.concatenate(preds_all, 0)
        variant_preds[model_name] = preds
        
        for rname, rmask in region_masks.items():
            rows.append(dict(
                week=week, model=model_name, region=rname,
                fss=regional_fss(preds, truth_concat, rmask, land2d, p66_map, window=5),
                pod=regional_pod(preds, truth_concat, rmask, land2d, p66_map),
                mae=regional_mae(preds, truth_concat, rmask, land2d),
                rmse=regional_rmse(preds, truth_concat, rmask, land2d),
                bias=regional_bias(preds, truth_concat, rmask, land2d),
                final_train_loss=float(np.mean(tr_losses)), final_val_loss=float(np.mean(va_losses)), mean_stop_epoch=float(np.mean(stop_eps)),
                allindia_ssim_mean=float(np.nanmean(pf_ssim)) if len(pf_ssim) else np.nan,
                allindia_nmae_mean=float(np.nanmean(pf_nmae)) if len(pf_nmae) else np.nan,
                allindia_mae_mean=float(np.nanmean(pf_mae)), allindia_mae_std=float(np.nanstd(pf_mae)),
                allindia_rmse_mean=float(np.nanmean(pf_rmse)), allindia_rmse_std=float(np.nanstd(pf_rmse)),
                n_folds=len(pf_rmse),
            ))
        ai = next(r for r in rows if r['model'] == model_name and r['region'] == 'AllIndia')
        print(f"    All-India Summary -> FSS: {ai['fss']:.3f} | MAE: {ai['mae']:.2f}")

    pd.DataFrame(rows).to_csv(out_csv(week), index=False)
    
    # Statistical Significance (Bootstrap)
    runnable = [p for p in CONTRAST_PAIRS if p[0] in variant_preds and p[1] in variant_preds]
    if runnable:
        print(f"\n  Bootstrapping contrasts utilizing {N_JOBS} CPUs...")
        crows = compute_contrasts(variant_preds, truth_concat, season_concat, region_masks, land2d, p66_map, pairs=runnable)
        for r in crows: r['week'] = week
        pd.DataFrame(crows).to_csv(out_contrast(week), index=False)
        print("  Contrasts saved.")

# 8. Dashboard Combination
def combine_and_plot():
    import glob
    mfiles = sorted(glob.glob("ablation_metrics_LTYO_W*.csv"))
    if not mfiles: print("No ablation metric files found."); return
        
    df = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True)
    weeks = sorted(df['week'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    
    print("\nGenerating All-India Multi-Metric Dashboard...")
    fig_d, axes_d = plt.subplots(1, 4, figsize=(22, 5.5), sharex=True)
    ai_df = df[df['region'] == 'AllIndia']
    
    metrics_titles = zip(['fss', 'pod', 'mae', 'rmse'], 
                         ['Fractions Skill Score (FSS)', 'Probability of Detection (p66)', 'Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)'])
    
    for ax, (metric, title) in zip(axes_d, metrics_titles):
        for mi, m in enumerate(MODELS):
            ys = [ai_df[(ai_df['week'] == w) & (ai_df['model'] == m)][metric].mean() for w in weeks]
            lw = 3.5 if m in ('M5', 'M5_noCBAM') else 1.8
            ax.plot(weeks, ys, '-o', color=colors[mi], label=m, linewidth=lw, markersize=6)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(weeks); ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlabel("Lead Week", fontsize=14, fontweight='bold')
        ax.set_ylabel("mm/week" if metric in ['mae', 'rmse'] else "Score", fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    h, l = axes_d[0].get_legend_handles_labels()
    fig_d.legend(h, l, loc='lower center', ncol=len(MODELS), fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.1))
    plt.suptitle("Architectural Ablation: Structural vs. Pixel Error Metrics", fontsize=20, fontweight='bold', y=1.08)
    
    plt.savefig("Ablation_MultiMetric_Dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if args.combine:
        combine_and_plot()
    elif args.week is not None:
        run_ablation(args.week)
    else:
        parser.error("Please provide --week {1,2,3,4} to train, or --combine to merge results.")