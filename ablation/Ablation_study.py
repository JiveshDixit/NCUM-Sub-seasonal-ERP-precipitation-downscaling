"""
architectural ablation study (m0 to m5 + m5_nocbam) for all 4 lead weeks.
this script breaks down the m5 model into smaller variants to see what 
each component actually contributes to the final skill.

design approach:
1. uses the exact same data pipeline, folds, and loss functions as the main script.
2. uses fixed hyperparameters across all models so we test architecture, not capacity.

evaluation:
- looks at mae, rmse, and bias in plain mm/week.
- uses a paired seasonal block bootstrap to see if differences between models 
  (like m4 vs m3, or with/without cbam) are statistically significant.

to run:
for w in 1 2 3 4; do python ablation_study.py --week $w & done; wait
python ablation_study.py --combine
"""

import os
import argparse
import warnings
import logging
warnings.filterwarnings('ignore')

# 1. argparse and gpu setup (needs to happen before tf loads)
parser = argparse.ArgumentParser()
parser.add_argument('--week', type=int, choices=[1, 2, 3, 4], help='Lead week (1-4)')
parser.add_argument('--combine', action='store_true',
                    help='Skip training; merge per-week CSVs into figures + contrast table.')
args, _unknown = parser.parse_known_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# optimization settings
SEED = 42
USE_MIXED_PRECISION = False   
USE_XLA = False               

# 2. bind gpu, then import the main training script
import tensorflow as tf

if not args.combine and args.week is not None:
    gpus = tf.config.list_physical_devices('GPU')
    gpu_idx = args.week - 1
    if gpus and gpu_idx < len(gpus):
        try:
            tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
            print(f"--- ablation week {args.week} bound to gpu {gpu_idx}: {gpus[gpu_idx].name} ---")
        except RuntimeError as e:
            print(e)

tf.config.threading.set_intra_op_parallelism_threads(24)
tf.config.threading.set_inter_op_parallelism_threads(24)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import random
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# importing custom layers and losses from the main pipeline
import Precipitation_correction_sub_seasonal_imd_qdm_no_blending as tp
from Precipitation_correction_sub_seasonal_imd_qdm_no_blending import (
    CombinedLoss, CBAMBlock, ResBlock, TakeLastStep, MatchShapes,
    neighborhood_mae, tweedie_metric_monitor, ssim_metric_monitor,
)
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
import geopandas as gpd
import regionmask

if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("mixed precision enabled.")

# 3. configuration paths and variables
tp.PREDICTOR_DIR   = "/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/erfgc2/final_weekly_timeseries"
tp.TARGET_DIR      = os.path.expanduser("~/hindcast_erp/Obs_precip")
tp.OROGRAPHY_FILE  = "/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/geopotential.nc"
tp.INDIA_SHP_PATH  = "/home/ncmrwf/bcwc/jivesh/Shape_files_India/India_State_Boundary_Updated.shp"
tp.XGB_RESULTS_DIR = "./../new/FEATURE_IMPORTANCE_RESULTS_ROBUST_REDUNDANCY_True"

PREDICTOR_DIR  = tp.PREDICTOR_DIR
TARGET_DIR     = tp.TARGET_DIR
INDIA_SHP_PATH = tp.INDIA_SHP_PATH

SEQUENCE_LENGTH = 6
BATCH_SIZE      = 8
PREDICT_BATCH   = 32

# fixed hps for all variants to keep the test fair
LR          = 3e-4
DROPOUT     = 0.1
LSTM_UNITS  = 256
BASE_FILTER = 128

ABL_EPOCHS   = 120
ABL_PATIENCE = 15

MODELS      = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M5_noCBAM']
LTYO_BLOCK  = 3
PLOT_METRIC = 'mae'   

# which model pairs to test against each other for significance
CONTRAST_PAIRS = [('M3', 'M4'), ('M5_noCBAM', 'M5')]
BOOT_N     = 1000
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
def out_fig(tag):        return f"ablation_{PLOT_METRIC}_decay_LTYO_{tag}"


# 4. ablation variant builder (the only unique logic in this script)
def build_variant(input_shape, model_name, loss_obj,
                  lr=LR, dropout=DROPOUT, lstm_units=LSTM_UNITS, base_filter=BASE_FILTER):
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

    # encoder
    x = layers.TimeDistributed(layers.Conv2D(f1, 3, padding='same'))(inputs)
    x = block(x, f1, td=True); enc1 = x
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(layers.Conv2D(f2, 3, padding='same'))(x)
    x = block(x, f2, td=True); enc2 = x
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    x = layers.TimeDistributed(layers.Conv2D(f3, 3, padding='same'))(x)
    x = block(x, f3, td=True)

    # bottleneck
    if use_lstm:
        x = layers.ConvLSTM2D(lstm_units, 3, padding='same', return_sequences=False)(x)
    else:
        x = TakeLastStep()(x)
        x = layers.Conv2D(lstm_units, 3, padding='same', activation='relu')(x)
    x = layers.SpatialDropout2D(dropout)(x)
    if use_cbam: x = CBAMBlock(lstm_units)(x)

    # decoder
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


# 5. cross-validation and metric helpers
def split_LTYO_embargo(years_per_sample, block=LTYO_BLOCK):
    unique_years = np.unique(years_per_sample)
    folds = []
    for start in range(0, len(unique_years), block):
        test_years = unique_years[start:start + block]
        tr_m, va_m, te_m, val_year, emb = tp.embargo_blocked_fold(
            years_per_sample, unique_years, test_years, embargo=1)
        tr, va, te = np.where(tr_m)[0], np.where(va_m)[0], np.where(te_m)[0]
        if len(te) == 0:
            continue
        print(f"    fold: test={list(test_years)} val={val_year} embargo={emb} "
              f"-> tr={len(tr)} va={len(va)} te={len(te)}")
        folds.append((tr, va, te))
    return folds

def region_mask(lats, lons, box):
    la0, la1, lo0, lo1 = box
    return ((lats[:, None] >= la0) & (lats[:, None] <= la1) &
            (lons[None, :] >= lo0) & (lons[None, :] <= lo1))

def _err_on(pred, truth, valid):           
    return (pred - truth)[:, valid]

def regional_rmse(pred, truth, rmask, land2d):
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    e = _err_on(pred, truth, valid)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.sqrt(np.nanmean(e ** 2)))

def regional_mae(pred, truth, rmask, land2d):
    """plain, unpooled mae in mm/week."""
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    e = _err_on(pred, truth, valid)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(np.abs(e)))

def regional_bias(pred, truth, rmask, land2d):
    valid = rmask & land2d
    if not valid.sum(): return np.nan
    e = _err_on(pred, truth, valid)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(e))


# 6. paired seasonal block bootstrap for statistical significance
def _metric_from_err(e2d, kind):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if kind == 'rmse': return float(np.sqrt(np.nanmean(e2d ** 2)))
        if kind == 'mae':  return float(np.nanmean(np.abs(e2d)))
        return float(np.nanmean(e2d))   

def compute_contrasts(variant_preds, truth, season_ids, region_masks, land2d,
                      pairs=CONTRAST_PAIRS, metrics=('mae', 'rmse', 'bias'),
                      n_boot=BOOT_N, alpha=BOOT_ALPHA, seed=SEED):
    """
    resamples whole seasons with replacement to see if the metric difference
    between two variants is significant (ci doesn't cross zero).
    """
    involved = sorted({x for pair in pairs for x in pair})
    vmasks = {r: (region_masks[r] & land2d) for r in region_masks}
    
    # pre-extract error arrays to save time
    err = {(name, r): _err_on(variant_preds[name], truth, vmasks[r])
           for name in involved for r in region_masks}

    seasons = np.unique(season_ids)
    by_season = {s: np.where(season_ids == s)[0] for s in seasons}
    rng = np.random.default_rng(seed)

    acc = {}
    for (A, B) in pairs:
        for r in region_masks:
            for mk in metrics:
                pt = _metric_from_err(err[(B, r)], mk) - _metric_from_err(err[(A, r)], mk)
                acc[(r, A, B, mk)] = {'point': pt, 'boot': np.empty(n_boot, dtype=float)}

    for b in range(n_boot):
        drawn = rng.choice(seasons, size=len(seasons), replace=True)
        sel = np.concatenate([by_season[s] for s in drawn])
        for (A, B) in pairs:
            for r in region_masks:
                eA, eB = err[(A, r)][sel], err[(B, r)][sel]
                for mk in metrics:
                    acc[(r, A, B, mk)]['boot'][b] = _metric_from_err(eB, mk) - _metric_from_err(eA, mk)

    rows = []
    for (r, A, B, mk), v in acc.items():
        lo = float(np.percentile(v['boot'], 100 * alpha / 2))
        hi = float(np.percentile(v['boot'], 100 * (1 - alpha / 2)))
        rows.append(dict(region=r, pair=f"{A}->{B}", metric=mk,
                         delta=v['point'], ci_lo=lo, ci_hi=hi,
                         significant=bool(lo > 0 or hi < 0)))
    return rows


# 7. main training and evaluation loop (runs one week at a time)
def run_ablation(week):
    print(f"\n--- running lead week {week} ---")
    p_fp = os.path.join(PREDICTOR_DIR, f"Week_{week}_AllYears.nc")
    t_fp = os.path.join(TARGET_DIR, f"IMD_week{week}_sum_25km_on_model_t.nc")

    X_da, Y_da = tp.load_and_preprocess(p_fp, t_fp, week)
    X_seq, Y_seq, _raw_da, Ytrue_da = tp.create_jjas_sequences(X_da, Y_da, Y_da, SEQUENCE_LENGTH)
    if len(X_seq) == 0:
        print("  no valid sequences found; aborting."); return

    years_filtered = pd.to_datetime(Ytrue_da.t.values).year.values
    lats, lons = X_da.latitude.values, X_da.longitude.values

    if os.path.exists(INDIA_SHP_PATH):
        gdf = gpd.read_file(INDIA_SHP_PATH)
        land2d = ~np.isnan(regionmask.Regions(gdf.geometry).mask(lons, lats).values)
    else:
        land2d = np.ones((len(lats), len(lons)), bool)
    region_masks = {name: region_mask(lats, lons, box) for name, box in REGION_BOXES.items()}

    folds = split_LTYO_embargo(years_filtered, LTYO_BLOCK)
    print(f"  cross validation: {len(folds)} folds generated")

    truth_concat  = np.concatenate([Y_seq[te].squeeze(-1) for (_, _, te) in folds], 0)
    season_concat = np.concatenate([years_filtered[te]    for (_, _, te) in folds], 0)

    shared_loss = CombinedLoss(w_tweedie=0.15, w_mae=1.0, w_ssim=15.0)

    rows = []
    variant_preds = {}            
    for model_name in MODELS:
        print(f"\n  --- training {model_name} ---")
        preds_all = []
        pf_rmse, pf_mae = [], []
        tr_losses, va_losses, stop_eps = [], [], []

        for fi, (tr, va, te) in enumerate(folds):
            mu = X_seq[tr].mean(axis=(0, 1, 2, 3))
            sd = X_seq[tr].std(axis=(0, 1, 2, 3)) + 1e-7
            X_trn = np.nan_to_num((X_seq[tr] - mu) / sd)
            X_van = np.nan_to_num((X_seq[va] - mu) / sd)
            X_ten = np.nan_to_num((X_seq[te] - mu) / sd)

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

            p = model.predict(X_ten, batch_size=PREDICT_BATCH, verbose=0).squeeze(-1)
            t = Y_seq[te].squeeze(-1)
            preds_all.append(p)
            pf_rmse.append(regional_rmse(p, t, region_masks['AllIndia'], land2d))
            pf_mae.append(regional_mae(p, t, region_masks['AllIndia'], land2d))
            print(f"    fold {fi+1}/{len(folds)} completed: stopped at epoch {stop_eps[-1]}, val_loss={va_losses[-1]:.4f}")
            del model; gc.collect()

        if not preds_all:
            continue
        preds = np.concatenate(preds_all, 0)
        variant_preds[model_name] = preds
        mtl, mvl, mep = float(np.mean(tr_losses)), float(np.mean(va_losses)), float(np.mean(stop_eps))

        for rname, rmask in region_masks.items():
            rows.append(dict(
                week=week, model=model_name, region=rname,
                mae=regional_mae(preds, truth_concat, rmask, land2d),
                rmse=regional_rmse(preds, truth_concat, rmask, land2d),
                bias=regional_bias(preds, truth_concat, rmask, land2d),
                final_train_loss=mtl, final_val_loss=mvl, mean_stop_epoch=mep,
                allindia_mae_mean=float(np.nanmean(pf_mae)),
                allindia_mae_std=float(np.nanstd(pf_mae)),
                allindia_rmse_mean=float(np.nanmean(pf_rmse)),
                allindia_rmse_std=float(np.nanstd(pf_rmse)),
                n_folds=len(pf_rmse),
            ))
        ai = next(r for r in rows if r['model'] == model_name and r['region'] == 'AllIndia')
        print(f"    all india summary -> mae: {ai['mae']:.2f} | rmse: {ai['rmse']:.2f} | bias: {ai['bias']:.2f}")

    pd.DataFrame(rows).to_csv(out_csv(week), index=False)
    print(f"\nsaved metrics to {out_csv(week)}")

    # bootrstrap testing for significance
    runnable = [p for p in CONTRAST_PAIRS if p[0] in variant_preds and p[1] in variant_preds]
    if runnable:
        print(f"\n  bootstrapping contrasts for {runnable} using {BOOT_N} resamples...")
        crows = compute_contrasts(variant_preds, truth_concat, season_concat,
                                  region_masks, land2d, pairs=runnable)
        for r in crows:
            r['week'] = week
        cdf = pd.DataFrame(crows)
        cdf.to_csv(out_contrast(week), index=False)
        
        ai_c = cdf[cdf['region'] == 'AllIndia']
        print("\n  --- all india contrasts (90% bootstrap interval) ---")
        for _, r in ai_c.iterrows():
            flag = "significant" if r['significant'] else "not significant (ci spans 0)"
            print(f"    {r['pair']:<16} {r['metric']:<5} "
                  f"diff={r['delta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  -> {flag}")


# 8. combine per-week results into the final decay figure and tables
def combine_and_plot():
    import glob
    mfiles = sorted(glob.glob("ablation_metrics_LTYO_W*.csv"))
    if not mfiles:
        print("no ablation metric files found. please run the per-week jobs first."); return
        
    df = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True)
    weeks = sorted(df['week'].unique())
    regions = list(REGION_BOXES.keys())
    metric, ylab = PLOT_METRIC, f"{PLOT_METRIC.upper()} (mm/week)"

    ncol = 4; nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True)
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    
    for ax, rname in zip(axes, regions):
        sub = df[df['region'] == rname]
        for mi, m in enumerate(MODELS):
            ys = [sub[(sub['week'] == w) & (sub['model'] == m)][metric].mean() for w in weeks]
            lw = 3 if m in ('M5', 'M5_noCBAM') else 1.8
            ax.plot(weeks, ys, '-o', color=colors[mi], label=m, linewidth=lw, markersize=5)
        ax.set_title(rname, fontsize=13, fontweight='bold')
        ax.set_xticks(weeks); ax.grid(alpha=0.2)
        ax.set_xlabel("Lead Week"); ax.set_ylabel(ylab)
        
    for ax in axes[len(regions):]:
        ax.axis('off')
        
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(MODELS), fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"Architectural Ablation: {PLOT_METRIC.upper()} Decay vs Lead Week", 
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_fig('ALL') + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_fig('ALL') + '.pdf', bbox_inches='tight')
    print(f"saved summary figures to {out_fig('ALL')}")

    # generating summary tables
    ai = df[df['region'] == 'AllIndia']
    summ = ai.groupby('model')[['mae', 'rmse', 'bias']].mean().reindex(MODELS)
    print("\n--- all india mean skill by model variant ---")
    print(summ.round(3).to_string())

    cfiles = sorted(glob.glob("ablation_contrasts_LTYO_W*_128_units.csv"))
    if cfiles:
        cdf = pd.concat([pd.read_csv(f) for f in cfiles], ignore_index=True)
        cdf.to_csv("ablation_contrasts_ALL_128_units.csv", index=False)
        ai_c = cdf[(cdf['region'] == 'AllIndia') & (cdf['metric'].isin(['mae', 'rmse']))]
        print("\n--- all india statistical contrasts per lead week ---")
        print("  pair             metric  week     delta        90% ci             resolved")
        for _, r in ai_c.sort_values(['pair', 'metric', 'week']).iterrows():
            print(f"  {r['pair']:<16} {r['metric']:<6} W{int(r['week'])}   "
                  f"{r['delta']:+.3f}   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]   "
                  f"{'YES' if r['significant'] else 'no'}")
        print("\nnote: confidence intervals are per week. do not average them across weeks.")
    else:
        print("\n(no contrast files found. you may need to run the bootstrap step.)")

if __name__ == "__main__":
    if args.combine:
        combine_and_plot()
    elif args.week is not None:
        run_ablation(args.week)
    else:
        parser.error("please provide --week {1,2,3,4} to train, or --combine to merge results.")