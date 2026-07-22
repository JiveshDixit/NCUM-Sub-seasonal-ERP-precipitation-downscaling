import os
import argparse

# ==============================================================================
# 0. ARGPARSE & GPU ISOLATION (MUST BE AT THE VERY TOP)
# ==============================================================================
parser = argparse.ArgumentParser(description='Run M5 Pipeline for a specific lead week.')
parser.add_argument('--week', type=int, choices=[1, 2, 3, 4], default=1, 
                    help='Specify a single lead week to process (1, 2, 3, or 4).')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed. 42 = PRODUCTION (unsuffixed output, Hyperband '
                         'runs). Any other value = REPLICATE: writes _SEED{n} files and '
                         'REUSES the production hyperparameters instead of re-tuning.')
parser.add_argument('--control', action='store_true',
                    help='SUBSET-SENSITIVITY CONTROL ARM. Swaps the marginal predictors '
                         'for the next-ranked unselected ones, keeping the channel count '
                         'identical. Everything else (seed, hyperparameters, folds, loss) '
                         'is held fixed, so any skill difference is attributable to the '
                         'predictor subset alone. Writes _CONTROL files.')
parser.add_argument('--control-swap', type=int, default=2, dest='control_swap',
                    help='Number of marginal predictors to swap out in the control arm '
                         '(default 2).')
args, unknown = parser.parse_known_args()

LEAD_WEEKS = [args.week]
SEED = args.seed

# A replicate MUST reuse the production hyperparameters. Hyperband is stochastic:
# re-tuning under a new seed would change the HPs as well as the initialization,
# confounding the very noise-floor measurement the replicate exists to make.
# The control arm MUST also reuse the production hyperparameters. If Hyperband were
# allowed to re-tune on the alternative subset, the two arms would differ in BOTH
# predictors and hyperparameters, and the experiment would no longer isolate the
# predictor effect it exists to measure.
CONTROL     = args.control
REUSE_HPS   = (SEED != 42) or CONTROL
SEED_SUFFIX = "" if SEED == 42 else f"_SEED{SEED}"
CONTROL_SUFFIX = "_CONTROL" if CONTROL else ""
OUT_SUFFIX  = f"{SEED_SUFFIX}{CONTROL_SUFFIX}"

gpu_idx = args.week - 1  # Maps Week 1 -> GPU 0, Week 2 -> GPU 1, etc.

print(f"\n--- BASH OVERRIDE: Running strictly for Lead Week {args.week} on GPU {gpu_idx} ---")
if REUSE_HPS:
    print(f"--- REPLICATE ARM: seed={SEED} | production HPs reused, NO tuning ---")
    print(f"--- Outputs suffixed '{SEED_SUFFIX}'. Production files will NOT be touched. ---")
elif not CONTROL:
    print(f"--- PRODUCTION ARM: seed=42 | Hyperband will run ---")
if CONTROL:
    print(f"--- SUBSET-SENSITIVITY CONTROL ARM ---")
    print(f"--- Swapping {args.control_swap} marginal predictors; seed, HPs, folds "
          f"and loss held FIXED. Outputs suffixed '{CONTROL_SUFFIX}'. ---")
print()

import site
sys_paths = site.getsitepackages()
if sys_paths:
    site_dir = sys_paths[0]
    os.environ['LD_LIBRARY_PATH'] = f"{site_dir}/nvidia/cudnn/lib:{site_dir}/nvidia/cublas/lib:" + os.environ.get('LD_LIBRARY_PATH', '')

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only see the specific GPU for this week
        tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
        print(f"Enabled memory growth. Bound to GPU: {gpus[gpu_idx].name}")
    except RuntimeError as e:
        print(e)

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
tf.keras.backend.set_floatx('float32')
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import gc
import json
import numpy as np
import xarray as xr
import pandas as pd
import keras_tuner as kt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
import geopandas as gpd
import regionmask
import random 

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# SEED now comes from --seed (set at the top). Do NOT reassign it here: a
# hard-coded 42 would clobber args.seed, every replicate would be an exact copy
# of production, and the measured noise floor would come out as ZERO.
#
# NOTE: seeding alone does NOT make GPU training deterministic -- cuDNN conv and
# ConvLSTM backward passes use non-deterministic atomic accumulation. That residual
# variability is exactly what the seed ensemble measures.
print(f'[SEED] Using seed {SEED}')
random.seed(SEED) 
np.random.seed(SEED) 
tf.random.set_seed(SEED)

PREDICTOR_DIR = os.path.expanduser("/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/erfgc2/final_weekly_timeseries")
OROGRAPHY_FILE = "/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/geopotential.nc"
INDIA_SHP_PATH = "/home/ncmrwf/bcwc/jivesh/Shape_files_India/India_State_Boundary_Updated.shp"
PREDICTOR_PATTERN = "Week_{week}_AllYears.nc"
TARGET_DIR = os.path.expanduser("~/hindcast_erp/Obs_precip")
TARGET_PATTERN = "IMD_week{week}_sum_25km_on_model_t.nc"

# Point this to where your XGBoost script saved the CSVs!
XGB_RESULTS_DIR = "./../new/FEATURE_IMPORTANCE_RESULTS_ROBUST_REDUNDANCY_False"

MODEL_DIR = "M5_Final_Models_oro_newer"
VALIDATION_DIR = "M5_Final_Results_oro_newer"

EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 8
SEQUENCE_LENGTH = 6

MSSSIM_H = None
MSSSIM_W = None

# ==============================================================================
# 2. DYNAMIC FEATURE SELECTION
# ==============================================================================

def get_robust_features(week, results_dir=XGB_RESULTS_DIR, cumulative_threshold=0.8):
    """
    Reads XGBoost permutation importance CSVs, applies strict statistical
    filtering, and explicitly forces downscaling anchor variables.
    """
    csv_path = os.path.join(results_dir, f"imp_week{week}.csv")
    base_features = ['raw_model_precip', 'orography_ht'] # Must be included for downscaling
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find feature importance file: {csv_path}. Run XGBoost script first.")

    df = pd.read_csv(csv_path)
    
    # Rule 1: Statistical Significance (Mean > 1 Std Dev)
    df_sig = df[df['perm_mean'] - df['perm_std'] > 0].copy()
    
    # Rule 2: Keep top features comprising 95% of the total predictive power
    df_sig = df_sig.sort_values('perm_mean', ascending=False)
    df_sig['importance_share'] = df_sig['perm_mean'] / df_sig['perm_mean'].sum()
    df_sig['cumulative_share'] = df_sig['importance_share'].cumsum()
    
    cutoff_idx = df_sig[df_sig['cumulative_share'] >= cumulative_threshold].index
    if len(cutoff_idx) > 0:
        first_breach = df_sig.index.get_loc(cutoff_idx[0])
        selected_features = df_sig.iloc[:first_breach + 1]['feature'].tolist()
    else:
        selected_features = df_sig['feature'].tolist()
        
    # Rule 3: Force append downscaling anchors if they got filtered out
    for feature in base_features:
        if feature not in selected_features:
            selected_features.append(feature)
            
    return selected_features


def get_control_features(week, results_dir=XGB_RESULTS_DIR, n_swap=2):
    """
    SUBSET-SENSITIVITY CONTROL ARM.

    Builds an ALTERNATIVE predictor subset of the SAME SIZE as the selected one, by
    swapping the n_swap lowest-importance selected predictors for the n_swap
    highest-importance predictors that the selection rule rejected.

    Design notes:
      * The channel count is preserved, so the network architecture is byte-identical
        between the two arms. Only the identity of the inputs changes.
      * The two downscaling anchors (raw_model_precip, orography_ht) are never swapped
        out. Removing the field being corrected, or the static terrain prior, would
        test something other than subset sensitivity.
      * The swap is deterministic and derived entirely from the archived importance
        table, so the control subset is exactly reproducible from the Zenodo archive.
    """
    csv_path = os.path.join(results_dir, f"imp_week{week}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find feature importance file: {csv_path}.")

    anchors  = ['raw_model_precip', 'orography_ht']
    selected = get_robust_features(week, results_dir=results_dir)

    df = pd.read_csv(csv_path).sort_values('perm_mean', ascending=False)
    ranked = df['feature'].tolist()

    # candidates to drop: selected, non-anchor, lowest importance first
    droppable = [f for f in reversed(ranked) if f in selected and f not in anchors]
    # candidates to add: never selected, highest importance first
    addable   = [f for f in ranked if f not in selected]

    k = min(n_swap, len(droppable), len(addable))
    if k == 0:
        raise RuntimeError(f"Week {week}: no valid swap available for the control arm.")

    drop = droppable[:k]
    add  = addable[:k]
    control = [f for f in selected if f not in drop] + add

    print(f"[CONTROL] Week {week}: swapping {k} marginal predictor(s)")
    print(f"[CONTROL]   dropped : {drop}")
    print(f"[CONTROL]   added   : {add}")
    print(f"[CONTROL]   selected subset ({len(selected)}): {selected}")
    print(f"[CONTROL]   control  subset ({len(control)}): {control}")
    assert len(control) == len(selected), \
        "Control arm must preserve channel count so the architecture is unchanged."
    return control

# ==============================================================================
# 3. CUSTOM LAYERS & LOSS FUNCTIONS (UNCHANGED)
# ==============================================================================
@register_keras_serializable()
def tweedie_calculation(y_true, y_pred, p):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    y_pred = tf.maximum(y_pred, 1e-5) 
    term_a = y_true * tf.pow(y_pred, 1.0 - p) / (1.0 - p)
    term_b = tf.pow(y_pred, 2.0 - p) / (2.0 - p)
    return tf.reduce_mean(term_b - term_a)

@register_keras_serializable()
def neighborhood_mae(y_true, y_pred, pool_size=3):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    valid_mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
    y_true_blur = tf.nn.avg_pool2d(y_true_clean, ksize=pool_size, strides=1, padding='SAME')
    y_pred_blur = tf.nn.avg_pool2d(y_pred, ksize=pool_size, strides=1, padding='SAME')
    abs_err = tf.abs(y_true_blur - y_pred_blur)
    denom = tf.reduce_mean(tf.abs(y_true_blur), axis=[1,2,3], keepdims=True) + 1.0
    rel_error = abs_err / denom
    masked_error = rel_error * valid_mask
    return tf.reduce_sum(masked_error) / (tf.reduce_sum(valid_mask) + 1e-6)


@register_keras_serializable()
def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index. 
    Forces the model to learn the 'clumpy' spatial structure of IMD data.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 1. Scrub the NaNs (Ocean mask)
    valid_mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
    
    # 2. Clean y_pred and zero out ocean predictions so they don't skew the score
    y_pred_clean = tf.where(tf.math.is_nan(y_pred), 0.0, y_pred)
    y_pred_clean = y_pred_clean * valid_mask
    
    # 3. Safely calculate the max value for the SSIM dynamic range
    max_val = tf.reduce_max(y_true_clean) + 1e-5
    
    # 4. Calculate SSIM
    ssim_val = tf.image.ssim(y_true_clean, y_pred_clean, max_val=max_val)
    
    # SAFETY: If a batch is 100% dry (all zeros), SSIM variance math can sometimes 
    # output NaN. We catch that here and return 1.0 (perfect match).
    ssim_val = tf.where(tf.math.is_nan(ssim_val), 1.0, ssim_val)
    
    return 1.0 - tf.reduce_mean(ssim_val)

@register_keras_serializable()
def ms_ssim_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    valid_mask   = tf.cast(~tf.math.is_nan(y_true), tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
    y_pred_clean = tf.where(tf.math.is_nan(y_pred), 0.0, y_pred)
    y_pred_clean = y_pred_clean * valid_mask

    y_true_clean = tf.image.resize_with_crop_or_pad(y_true_clean, MSSSIM_H, MSSSIM_W)
    y_pred_clean = tf.image.resize_with_crop_or_pad(y_pred_clean, MSSSIM_H, MSSSIM_W)

    max_val = tf.reduce_max(y_true_clean) + 1e-5
    ms = tf.image.ssim_multiscale(y_true_clean, y_pred_clean, max_val=max_val,
                                  power_factors=(0.071, 0.453, 0.476))
    ms = tf.where(tf.math.is_nan(ms), 1.0, ms)
    return 1.0 - tf.reduce_mean(ms)

@register_keras_serializable()
def ssim_metric_monitor(y_true, y_pred):
    return ssim_loss(y_true, y_pred)

@register_keras_serializable()
def tweedie_metric_monitor(y_true, y_pred):
    return tweedie_calculation(y_true, y_pred, p=1.65)

@register_keras_serializable()
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, w_tweedie=0.15, w_mae=1.0, w_ssim=15.0, tweedie_p=1.65, **kwargs):
        super().__init__(**kwargs)
        self.w_tweedie = w_tweedie
        self.w_mae = w_mae
        self.w_ssim = w_ssim 
        self.tweedie_p = tweedie_p

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        valid_mask = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y_true_clean = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        y_pred_clean = tf.where(tf.math.is_nan(y_pred), 0.0, y_pred)
        y_pred_clean = y_pred_clean * valid_mask

        t_loss = tweedie_calculation(y_true_clean, y_pred_clean, self.tweedie_p)
        m_loss = neighborhood_mae(y_true, y_pred_clean, pool_size=3) 
        s_loss = ssim_loss(y_true_clean, y_pred_clean)

        total_loss = (self.w_tweedie * t_loss) + (self.w_mae * m_loss) + (self.w_ssim * s_loss)
        return tf.where(tf.math.is_finite(total_loss), total_loss, tf.zeros_like(total_loss))

    def get_config(self):
        config = super().get_config()
        config.update({
            "w_tweedie": float(self.w_tweedie),
            "w_mae":     float(self.w_mae),
            "w_ssim":    float(self.w_ssim),
            "tweedie_p": self.tweedie_p
        })
        return config

@register_keras_serializable()
class TakeLastStep(layers.Layer):
    def call(self, x):
        return x[:, -1]

@register_keras_serializable()
class MatchShapes(layers.Layer):
    def call(self, inputs):
        x, ref = inputs
        return tf.image.resize_with_crop_or_pad(x, tf.shape(ref)[1], tf.shape(ref)[2])

class LossBreakdown(keras.callbacks.Callback):
    def __init__(self, w_tweedie=0.15, w_mae=1.0, w_ssim=15):
        super().__init__()
        self.w_tweedie = w_tweedie
        self.w_mae = w_mae
        self.w_ssim = w_ssim

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return
        t_val = logs.get('tweedie_metric_monitor', 0.0)
        m_val = logs.get('neighborhood_mae', 0.0) 
        s_val = logs.get('ssim_metric_monitor', 0.0)
        total_loss = logs.get('loss', 0.0)
        
        t_weighted = t_val * self.w_tweedie
        m_weighted = m_val * self.w_mae
        s_weighted = s_val * self.w_ssim
        
        print(f"\n--- Epoch {epoch+1} Loss Breakdown ---")
        print(f"  Total Loss:     {total_loss:.4f}")
        print(f"  Tweedie:        {t_val:.4f} (Weighted: {t_weighted:.4f})")
        print(f"  Neigh-MAE:      {m_val:.4f} (Weighted: {m_weighted:.4f})")
        print(f"  SSIM:           {s_val:.4f} (Weighted: {s_weighted:.4f})")
        print("--------------------------------------\n")

@register_keras_serializable()
class CBAMBlock(layers.Layer):
    def __init__(self, filters, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.mlp1 = layers.Dense(filters // ratio, activation='relu', use_bias=False)
        self.mlp2 = layers.Dense(filters, use_bias=False)
        self.conv_spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=False)

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        mlp_shape = (None, 1, 1, channel_dim)
        self.mlp1.build(mlp_shape)
        mlp1_out = self.mlp1.compute_output_shape(mlp_shape)
        self.mlp2.build(mlp1_out)
        spatial_shape = list(input_shape)
        spatial_shape[-1] = 2 
        self.conv_spatial.build(tuple(spatial_shape))
        super().build(input_shape)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        avg_out = self.mlp2(self.mlp1(avg_pool))
        max_out = self.mlp2(self.mlp1(max_pool))
        channel_att = tf.nn.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        avg_sp = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_sp = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_sp, max_sp], axis=-1)
        spatial_att = self.conv_spatial(concat)
        x = x * spatial_att
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'ratio': self.ratio})
        return config

@register_keras_serializable()
class ResBlock(layers.Layer):
    def __init__(self, filters, use_cbam=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_cbam = use_cbam
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.gn1 = layers.GroupNormalization(groups=8)
        self.act = layers.Activation(lambda x: x * tf.math.tanh(tf.math.softplus(x)))
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.gn2 = layers.GroupNormalization(groups=8)
        
        if self.use_cbam:
            self.cbam = CBAMBlock(filters)
        else:
            self.cbam = None
            
        self.add_layer = layers.Add()
        self.shortcut_conv = None

    def build(self, input_shape):
        self.conv1.build(input_shape)
        feature_shape = list(input_shape)
        feature_shape[-1] = self.filters
        feature_shape = tuple(feature_shape)
        self.gn1.build(feature_shape)
        self.conv2.build(feature_shape)
        self.gn2.build(feature_shape)
        if self.use_cbam:
            self.cbam.build(feature_shape)
        if input_shape[-1] != self.filters:
            if self.shortcut_conv is None:
                self.shortcut_conv = layers.Conv2D(self.filters, 1, padding='same')
            self.shortcut_conv.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        shortcut = x
        x = self.conv1(x); x = self.gn1(x); x = self.act(x)
        x = self.conv2(x); x = self.gn2(x)
        if self.cbam is not None: x = self.cbam(x)
        if self.shortcut_conv is not None: shortcut = self.shortcut_conv(shortcut)
        x = self.add_layer([x, shortcut])
        x = self.act(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'use_cbam': self.use_cbam})
        return config

# ==============================================================================
# 4. M5 MODEL ARCHITECTURE
# ==============================================================================

def build_M5_model(input_shape, lr=1e-4, dropout=0.0, lstm_units=256, base_filter=64, loss_obj=None):
    f1 = base_filter
    f2 = base_filter * 2
    f3 = base_filter * 4
    
    inputs = layers.Input(shape=input_shape) 
    
    # --- ENCODER ---
    x = layers.TimeDistributed(layers.Conv2D(f1, 3, padding='same'))(inputs)
    x = layers.TimeDistributed(ResBlock(f1, use_cbam=True))(x)
    enc1 = x
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    
    x = layers.TimeDistributed(layers.Conv2D(f2, 3, padding='same'))(x)
    x = layers.TimeDistributed(ResBlock(f2, use_cbam=True))(x)
    enc2 = x
    x = layers.TimeDistributed(layers.MaxPooling2D())(x)
    
    x = layers.TimeDistributed(layers.Conv2D(f3, 3, padding='same'))(x)
    x = layers.TimeDistributed(ResBlock(f3, use_cbam=True))(x)
    
    # --- BOTTLENECK ---
    x = layers.ConvLSTM2D(lstm_units, 3, padding='same', return_sequences=False)(x)
    x = layers.SpatialDropout2D(dropout)(x)
    x = CBAMBlock(lstm_units)(x)
    
    # --- DECODER ---
    x = layers.UpSampling2D()(x)
    s2 = TakeLastStep()(enc2)
    x = layers.Concatenate()([MatchShapes()([x, s2]), s2])
    x = ResBlock(f2, use_cbam=True)(x)
    x = layers.SpatialDropout2D(dropout)(x)
    
    x = layers.UpSampling2D()(x)
    s1 = TakeLastStep()(enc1)
    x = layers.Concatenate()([MatchShapes()([x, s1]), s1])
    x = ResBlock(f1, use_cbam=True)(x)
    
    outputs = layers.Conv2D(1, 1, activation='softplus')(x)
    
    model = Model(inputs, outputs, name="M5_Deep_Recurrent_Attn")
    
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-3, clipnorm=1.0)
    
    if loss_obj is None:
        loss_obj = CombinedLoss(w_tweedie=0.15, w_mae=1.0, w_ssim=15.0)
    
    model.compile(
        optimizer=opt,
        loss=loss_obj,
        metrics=[neighborhood_mae, tweedie_metric_monitor, ssim_metric_monitor]
    )
    return model

class M5HyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        lr = hp.Float('lr', 1e-4, 1e-3, sampling='log')
        dropout = hp.Choice('dropout', [0.0, 0.1, 0.2, 0.3, 0.4])
        lstm_units = hp.Choice('lstm_units', [32, 64, 128, 256])
        base_filter = hp.Choice('base_filter', [16, 32, 64, 128])

        model = build_M5_model(
            self.input_shape, 
            lr=lr, 
            dropout=dropout,
            lstm_units=lstm_units,
            base_filter=base_filter
        )
        dummy = tf.zeros((1,) + self.input_shape)
        model(dummy, training=False)
        return model

# ==============================================================================
# 5. DATA PROCESSING
# ==============================================================================
def apply_shapefile_mask(xr_dataset, shapefile_path):
    print(f"  Applying spatial mask from: {os.path.basename(shapefile_path)}")
    gdf = gpd.read_file(shapefile_path)
    mask = regionmask.mask_geopandas(gdf, xr_dataset.longitude, xr_dataset.latitude)
    is_inside = ~np.isnan(mask)
    return xr_dataset.where(is_inside)

def calculate_ivt(ds):
    g = 9.80665
    if not all(var in ds for var in ['q', 'u', 'v']): return None
    try:
        p_coord = None
        for coord in ['p', 'level', 'isobaricInhPa', 'pressure']:
            if coord in ds.coords:
                p_coord = coord
                break
        if p_coord is None: return None
        p_vals = ds[p_coord]
        p_factor = 100.0 if p_vals.max() < 2000 else 1.0

        qu_int = (ds['q'] * ds['u']).integrate(coord=p_coord) * (p_factor / g)
        qv_int = (ds['q'] * ds['v']).integrate(coord=p_coord) * (p_factor / g)
        return np.sqrt(qu_int**2 + qv_int**2)
    except Exception:
        return None

def load_and_preprocess(p_fp, t_fp, week):
    print(f"Loading: {os.path.basename(p_fp)}")

    # DYNAMIC SELECTION REPLACES HARDCODING
    # print('week1 features: ', {get_robust_features('1')})
    if CONTROL:
        SELECTED_FEATURES = get_control_features(week, n_swap=args.control_swap)
    else:
        SELECTED_FEATURES = get_robust_features(week)
    print(f"\n--- Features Dynamically Selected for Week {week} ---")
    print(f"{SELECTED_FEATURES}\n")

    def standardize_coords(ds_or_da):
        rename_map = {}
        if 'lat' in ds_or_da.coords: rename_map['lat'] = 'latitude'
        if 'lon' in ds_or_da.coords: rename_map['lon'] = 'longitude'
        return ds_or_da.rename(rename_map) if rename_map else ds_or_da

    def open_nc(path):
        for eng in ['netcdf4', 'h5netcdf', 'scipy']:
            try: return xr.open_dataset(path, engine=eng, decode_times=True)
            except: continue
        raise IOError(f"Could not open {path}")

    p_ds = standardize_coords(open_nc(p_fp))
    t_ds = standardize_coords(open_nc(t_fp))

    Y_obs = t_ds['tp_weekly_sum']
    if Y_obs.max() < 10.0: Y_obs = Y_obs * 1000.0
    if os.path.exists(INDIA_SHP_PATH):
        Y_obs = apply_shapefile_mask(Y_obs, INDIA_SHP_PATH)

    p_dates = pd.to_datetime(p_ds.t.values).normalize()
    y_dates = pd.to_datetime(Y_obs.t.values).normalize()
    
    df_p = pd.DataFrame({'p_time': p_dates, 'p_idx': np.arange(len(p_dates))}).sort_values('p_time')
    df_y = pd.DataFrame({'y_time': y_dates, 'y_idx': np.arange(len(y_dates))}).sort_values('y_time')
    
    merged = pd.merge_asof(df_y, df_p, left_on='y_time', right_on='p_time', 
                           direction='nearest', tolerance=pd.Timedelta('3d')).dropna()
    
    p_ds  = p_ds.isel(t=merged['p_idx'].values.astype(int))
    Y_obs = Y_obs.isel(t=merged['y_idx'].values.astype(int))
    p_ds  = p_ds.assign_coords(t=Y_obs.t)
    
    X_list = []
    def add(da, name):
        if da is not None:
            da = standardize_coords(da)
            for dim in ['surface', 'ht', 'valid_time', 'number', 'expver', 'toa']:
                if dim in da.dims: da = da.squeeze(dim, drop=True)
            if 't' not in da.dims: da, _ = xr.broadcast(da, p_ds['t'])
            if 'latitude' in da.dims and 'longitude' in da.dims:
                X_list.append(da.fillna(0.0).assign_coords(channel=name))

    add(p_ds['tot_precip'], 'raw_model_precip')

    if 'q' in p_ds:
        is_pa = p_ds.p.max() > 2000
        for lev in [925, 850, 500]:
            try: add(p_ds['q'].sel(p=lev * 100.0 if is_pa else lev, method='nearest').drop_vars('p'), f'q_{lev}')
            except: pass
            
    if 'temp' in p_ds:
        try: 
            t = p_ds['temp'].sel(ht=1.5, method='nearest')
            if 'ht' in t.coords: t = t.drop_vars('ht')
            add(t, 't_1p5')
        except: pass

    if 'sm' in p_ds:
        sm = p_ds['sm']
        if 'level6' in sm.dims: sm = sm.mean('level6')
        add(sm, 'sm')

    for lev in [850, 200]:
        is_pa = p_ds.p.max() > 2000
        target = lev * 100.0 if is_pa else lev
        if 'u' in p_ds:
            try: add(p_ds['u'].sel(p=target, method='nearest').drop_vars('p'), f'u_{lev}')
            except: pass
        if 'v' in p_ds:
            try: add(p_ds['v'].sel(p=target, method='nearest').drop_vars('p'), f'v_{lev}')
            except: pass

    try:
        lat_key = 'latitude' if 'latitude' in p_ds.coords else 'lat'
        lon_key = 'longitude' if 'longitude' in p_ds.coords else 'lon'
        dy = 111000.0; lat_rad = np.deg2rad(p_ds.coords[lat_key])
        dx = 111000.0 * np.cos(lat_rad)
        tp = 85000 if p_ds.p.max() > 2000 else 850
        q = p_ds['q'].sel(p=tp, method='nearest')
        u = p_ds['u'].sel(p=tp, method='nearest')
        v = p_ds['v'].sel(p=tp, method='nearest')
        div_flux = ((q*u).differentiate(lon_key) / dx) + ((q*v).differentiate(lat_key) / dy)
        add((-div_flux).assign_coords(channel='mfc_850'), 'mfc_850')
    except: pass

    if 'ht_1' in p_ds:
        is_pa = p_ds.p.max() > 2000
        try: add(p_ds['ht_1'].sel(p=500 * 100.0 if is_pa else 500, method='nearest').drop_vars('p'), 'z_500')
        except: pass

    if 'olr' in p_ds: add(p_ds['olr'], 'olr')

    if 't' in p_ds.coords:
        month = p_ds.t.dt.month
        c_grid, _ = xr.broadcast(np.cos(2 * np.pi * month / 12.0), p_ds['tot_precip'].isel(t=0).squeeze())
        s_grid, _ = xr.broadcast(np.sin(2 * np.pi * month / 12.0), p_ds['tot_precip'].isel(t=0).squeeze())
        add(c_grid, 'month_cos'); add(s_grid, 'month_sin')

    if os.path.exists(OROGRAPHY_FILE):
        try:
            orog_ds = open_nc(OROGRAPHY_FILE)
            orog = orog_ds['z'] / 9.80665 if 'z' in orog_ds else (orog_ds['ht'] if 'ht' in orog_ds else orog_ds['hgt'])
            if 'valid_time' in orog.dims: orog = orog.isel(valid_time=0).squeeze()
            orog = standardize_coords(orog).interp_like(standardize_coords(p_ds['tot_precip'].isel(t=0).squeeze()), method='nearest', kwargs={'fill_value': 'extrapolate'})
            orog_3d, _ = xr.broadcast(orog, p_ds['tot_precip'])
            add(orog_3d, 'orography_ht')
        except: pass

    X = xr.concat(X_list, dim='channel').transpose('t', 'channel', 'latitude', 'longitude')
    
    available = [f for f in SELECTED_FEATURES if f in X.channel.values]
    X = X.sel(channel=available)
    
    print(f"\n FINAL PREDICTORS ({len(X.channel)}): {X.channel.values.tolist()} \n")
    
    X = X.interp_like(Y_obs, method='linear', kwargs={'fill_value': 'extrapolate'})
    X, Y_obs = xr.align(X, Y_obs, join='inner')
    
    return X.fillna(0.0), Y_obs


def create_jjas_sequences(X, Y_target, Y_true, seq_length):
    X_seq, Y_seq, Raw_seq, Y_true_seq, valid_times = [], [], [], [], []
    times = pd.to_datetime(Y_target.t.values)
    print("  Filtering sequences for JJAS...")
    
    Raw_Precip_Full = X.sel(channel='raw_model_precip')

    for i in range(seq_length, len(times)):
        target_date = times[i]
        
        if target_date.month in [6, 7, 8, 9]:
            # Guard against the 9-month winter gap
            win = times[i - seq_length + 1 : i + 1]
            if (win[-1] - win[0]).days > (seq_length + 1) * 7:
                continue
                
            x_slice = np.transpose(X.isel(t=slice(i - seq_length + 1, i + 1)).values, (0, 2, 3, 1))
            y_val = Y_target.isel(t=i).values
            
            if x_slice.shape[0] == seq_length and not np.isnan(x_slice).any() and not np.isnan(y_val).all():
                X_seq.append(x_slice)
                Y_seq.append(y_val[..., np.newaxis]) 
                Raw_seq.append(Raw_Precip_Full.isel(t=i).values)
                Y_true_seq.append(Y_true.isel(t=i).values)
                valid_times.append(Y_target.t.values[i]) 

    if len(X_seq) == 0: 
        return np.array([]), np.array([]), None, None

    val_coords = {'t': valid_times, 'latitude': Y_target.latitude.values, 'longitude': Y_target.longitude.values}
    target_dims = ('t', 'latitude', 'longitude')
    
    return np.array(X_seq), np.array(Y_seq), \
           xr.DataArray(np.array(Raw_seq), coords=val_coords, dims=target_dims, name="raw_model_precip"), \
           xr.DataArray(np.array(Y_true_seq), coords=val_coords, dims=target_dims, name="imd_precip")


# ==============================================================================
# 6. PIPELINE FUNCTIONS (EMBARGO, QDM, EQM)
# ==============================================================================
def embargo_blocked_fold(years, unique_years, test_years, embargo=1, reserve_years=None):
    test_set = set(np.asarray(test_years).tolist())
    test_mask = np.isin(years, list(test_set))
 
    pool = np.array([y for y in unique_years if y not in test_set])
    test_centre = float(np.mean(test_years))
    val_year = int(pool[np.argmax(np.abs(pool - test_centre))])
    val_mask = (years == val_year)
 
    embargo_years = set()
    for ty in list(test_set) + [val_year]:
        for d in range(1, embargo + 1):
            embargo_years.add(ty - d); embargo_years.add(ty + d)
    embargo_years &= set(np.asarray(unique_years).tolist())
    embargo_years -= (test_set | {val_year})
 
    drop = set(embargo_years)
    if reserve_years is not None:
        drop |= (set(np.asarray(reserve_years).tolist()) - test_set - {val_year})
 
    train_mask = ~(test_mask | val_mask | np.isin(years, list(drop)))
    return train_mask, val_mask, test_mask, val_year, sorted(embargo_years)

def _pool_samples(arr3d, i, j, radius):
    H, W = arr3d.shape[1], arr3d.shape[2]
    i0, i1 = max(0, i - radius), min(H, i + radius + 1)
    j0, j1 = max(0, j - radius), min(W, j + radius + 1)
    b = arr3d[:, i0:i1, j0:j1].reshape(-1)
    return b[np.isfinite(b)]
 
def apply_qdm_pooled(obs_train, mod_train, mod_test, n_quantiles=100, pool_radius=1, drizzle_thresh=0.1):
    print(f"   -> Running pooled QDM (radius={pool_radius}, {n_quantiles} quantiles)...")
    out = np.copy(mod_test); T, H, W = mod_test.shape
    q = np.linspace(0, 100, n_quantiles); eps = 1e-6
    for i in range(H):
        for j in range(W):
            x = mod_test[:, i, j]
            if np.isnan(x).all(): continue
            o_pool = _pool_samples(obs_train, i, j, pool_radius)
            m_pool = _pool_samples(mod_train, i, j, pool_radius)
            if len(o_pool) < 20 or len(m_pool) < 20: continue
            o_q = np.percentile(o_pool, q); m_q = np.percentile(m_pool, q)
            m_uniq, idx = np.unique(m_q, return_index=True)
            if len(m_uniq) < 2: continue
            o_at = o_q[idx]; r = np.linspace(0, 1, len(m_uniq))
            tau = np.interp(x, m_uniq, r, left=0.0, right=1.0)
            ratio = np.interp(tau, r, o_at) / (np.interp(tau, r, m_uniq) + eps)
            fin = np.isfinite(ratio)
            if fin.any():
                lo, hi = np.percentile(ratio[fin], [1, 99]); ratio = np.clip(ratio, lo, hi)
            mapped = x * ratio
            mapped[x <= drizzle_thresh] = 0.0
            out[:, i, j] = mapped
    return out

def apply_eqm(obs_train, mod_train, mod_test):
    print("   -> Generating Conventional Simple EQM Baseline...")
    eqm_test = np.copy(mod_test)
    T, H, W = mod_test.shape
    quantiles = np.linspace(0, 100, 1000)
    for i in range(H):
        for j in range(W):
            o_tr, m_tr, m_te = obs_train[:, i, j], mod_train[:, i, j], mod_test[:, i, j]
            valid_mask = ~np.isnan(o_tr) & ~np.isnan(m_tr)
            if np.sum(valid_mask) < 20 or np.isnan(m_te).all(): continue
            o_valid, m_valid = o_tr[valid_mask], m_tr[valid_mask]
            m_q, o_q = np.percentile(m_valid, quantiles), np.percentile(o_valid, quantiles)
            m_uniq, u_idx = np.unique(m_q, return_index=True)
            mapped = np.interp(m_te, m_uniq, o_q[u_idx])
            mapped[m_te <= 0.1] = 0.0
            eqm_test[:, i, j] = mapped
    return eqm_test

def plot_recent_performance(validation_ds, week_num, num_weeks=8, output_dir="PLOTS"):
    os.makedirs(output_dir, exist_ok=True)
    subset = validation_ds.isel(t=slice(-num_weeks, None))
    times = pd.to_datetime(subset.t.values)
    lat_min, lat_max = subset.latitude.min(), subset.latitude.max()
    lon_min, lon_max = subset.longitude.min(), subset.longitude.max()
    
    levels = [0, 5, 10, 20, 50, 100, 150, 200, 300, 500]
    cmap_precip = plt.cm.gist_ncar_r
    norm_precip = BoundaryNorm(levels, ncolors=cmap_precip.N, clip=True)
    bias_levels = [-100, -50, -20, -10, -5, 5, 10, 20, 50, 100]
    cmap_bias = plt.cm.RdBu
    norm_bias = BoundaryNorm(bias_levels, ncolors=cmap_bias.N, extend='both')

    print(f"  Generating plots for the last {num_weeks} weeks...")
    for i, t in enumerate(times):
        date_str = t.strftime('%Y-%m-%d')
        obs = subset['imd_precip'].isel(t=i)
        pred_qdm = subset['m5_dl_qdm_precip'].isel(t=i)
        pred_dl = subset['dl_only_precip'].isel(t=i) # Added: Raw DL output
        raw = subset['raw_model_precip'].isel(t=i)
        
        # Calculate bias against the RAW Deep Learning output, not QDM
        bias_dl = pred_dl - obs 
        
        fig = plt.figure(figsize=(25, 5)) # Slightly wider for an extra panel
        proj = ccrs.PlateCarree()
        
        ax1 = fig.add_subplot(1, 5, 1, projection=proj)
        im1 = ax1.pcolormesh(obs.longitude, obs.latitude, obs, cmap=cmap_precip, norm=norm_precip, transform=proj)
        ax1.set_title(f"IMD Observed\nWeek {week_num} | {date_str}")
        
        ax2 = fig.add_subplot(1, 5, 2, projection=proj)
        ax2.pcolormesh(raw.longitude, raw.latitude, raw, cmap=cmap_precip, norm=norm_precip, transform=proj)
        ax2.set_title("Raw NCUM")
        
        ax3 = fig.add_subplot(1, 5, 3, projection=proj)
        ax3.pcolormesh(pred_dl.longitude, pred_dl.latitude, pred_dl, cmap=cmap_precip, norm=norm_precip, transform=proj)
        ax3.set_title("M5 DL (Before QDM)") # Check this for spatial smoothing/bias
        
        ax4 = fig.add_subplot(1, 5, 4, projection=proj)
        ax4.pcolormesh(pred_qdm.longitude, pred_qdm.latitude, pred_qdm, cmap=cmap_precip, norm=norm_precip, transform=proj)
        ax4.set_title("M5 DL + QDM")
        
        ax5 = fig.add_subplot(1, 5, 5, projection=proj)
        ax5.pcolormesh(bias_dl.longitude, bias_dl.latitude, bias_dl, cmap=cmap_bias, norm=norm_bias, transform=proj)
        ax5.set_title("Bias (DL Only - Obs)") # Visual confirmation of the -10 bias
        
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.coastlines()
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label="Precip (mm/week)")
        plt.savefig(os.path.join(output_dir, f"Week{week_num}_Forecast_{date_str}.png"), bbox_inches='tight', dpi=150)
        plt.close()

# ==============================================================================
# 7. MAIN EXECUTION LOOP (L3YO CROSS-VALIDATION)
# ==============================================================================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    GLOBAL_BATCH_SIZE = BATCH_SIZE

    for week in LEAD_WEEKS:
        print(f"\n{'='*50}\nProcessing Week: {week}\n{'='*50}")

        p_fp = os.path.join(PREDICTOR_DIR, PREDICTOR_PATTERN.format(week=week))
        t_fp = os.path.join(TARGET_DIR, TARGET_PATTERN.format(week=week))
        X, Y_obs = load_and_preprocess(p_fp, t_fp, week)
        
        X_seq_np, Y_seq_np, Raw_val_da, Y_true_da = create_jjas_sequences(X, Y_obs, Y_obs, SEQUENCE_LENGTH)
        
        print(f"  Total Samples Found: {len(X_seq_np)}")
        if len(X_seq_np) < 50:
            print("  WARNING: Dataset is suspiciously small. Check NaNs in raw data.")
            continue

        times = pd.DatetimeIndex(Y_true_da.t.values)
        years = times.year
        unique_years = np.unique(years)
        
        def make_train_dataset(x, y):
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            ds = ds.shuffle(1024)
            ds = ds.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
            return ds.repeat().prefetch(tf.data.AUTOTUNE)

        # ======================================================================
        # PHASE 1: HYPERPARAMETER TUNING 
        # ======================================================================
        print("\n--- PHASE 1: Tuning Optimal Architecture ---")

        # ======================================================================
        # START: PARAMETERS FOR MS:SSIM
        # ======================================================================        
        global MSSSIM_H, MSSSIM_W
        H_grid, W_grid = int(Y_obs.latitude.size), int(Y_obs.longitude.size)
        MSSSIM_H = (H_grid // 4) * 4
        MSSSIM_W = (W_grid // 4) * 4
        MIN_SIDE = 44  # two 2x downsamples must leave the coarsest scale > 11-px SSIM window
        if min(MSSSIM_H, MSSSIM_W) < MIN_SIDE:
            raise ValueError(
                f"Grid {H_grid}x{W_grid} too small for 3-scale MS-SSIM "
                f"(cropped {MSSSIM_H}x{MSSSIM_W}). Drop to 2 scales "
                "(power_factors=(0.3,0.7)) or use single-scale ssim_loss.")
        print(f"  MS-SSIM crop fixed at {MSSSIM_H}x{MSSSIM_W} (from grid {H_grid}x{W_grid})")

        # ======================================================================
        # end: PARAMETERS FOR MS:SSIM
        # ======================================================================
        chunk_size = 3
        year_blocks = [unique_years[i:i + chunk_size] for i in range(0, len(unique_years), chunk_size)]
        TUNE_VAL_YEARS = unique_years[-4:]
        
        tune_train_mask = years < unique_years[-4] 
        tune_val_mask = ~tune_train_mask
        
        X_tune_train, Y_tune_train = X_seq_np[tune_train_mask], Y_seq_np[tune_train_mask]
        X_tune_val, Y_tune_val = X_seq_np[tune_val_mask], Y_seq_np[tune_val_mask]
        
        X_t_mean, X_t_std = X_tune_train.mean(axis=(0,1,2,3)), X_tune_train.std(axis=(0,1,2,3))
        X_t_train_n = np.nan_to_num((X_tune_train - X_t_mean) / (X_t_std + 1e-7))
        X_t_val_n = np.nan_to_num((X_tune_val - X_t_mean) / (X_t_std + 1e-7))
        t_train_steps = len(X_t_train_n) // GLOBAL_BATCH_SIZE
        
        tuner = kt.Hyperband(
            M5HyperModel(X_seq_np.shape[1:]),
            objective='val_loss',
            max_epochs=25,
            factor=3,
            hyperband_iterations=1,
            directory='m5_tuning_final',
            project_name=f'week{week}',
            overwrite=False
        )
        
        if REUSE_HPS:
            # Replicate arm: read the production hyperparameters back. Do NOT search.
            # With overwrite=False an INCOMPLETE oracle would otherwise RESUME the
            # search under this seed and select different HPs -- silently confounding
            # the experiment. Fail loudly instead.
            if not tuner.oracle.get_best_trials(1):
                raise RuntimeError(
                    f"[ARM] No completed trials in m5_tuning_final/week{week}. "
                    f"Replicate arms must reuse the production run's hyperparameters. "
                    f"Run the production arm for this week first (no --seed).")
            best_hps = tuner.get_best_hyperparameters(1)[0]
            print(f"Reusing production HPs (NO SEARCH): LR={best_hps.get('lr')}, "
                  f"LSTM={best_hps.get('lstm_units')}")
        else:
            tuner.search(
                make_train_dataset(X_t_train_n, Y_tune_train), 
                validation_data=(X_t_val_n, Y_tune_val), 
                epochs=15, 
                verbose=1,
                steps_per_epoch=t_train_steps
            )
            best_hps = tuner.get_best_hyperparameters(1)[0]
            print(f"Optimal HPs Found: LR={best_hps.get('lr')}, LSTM={best_hps.get('lstm_units')}")

        # ======================================================================
        # PHASE 2: 3-YEAR BLOCK CROSS-VALIDATION
        # ======================================================================
        print("\n--- PHASE 2: 3-Year Block Cross Validation ---")
        all_test_times, all_truth_test, all_raw_test = [], [], []
        all_eqm_test, all_dl_test, all_dl_qdm_test = [], [], []

        for test_years in year_blocks:
            print(f"\n>> Training Fold: Holding out {test_years} for Testing")
            
            train_mask, val_mask, test_mask, val_year, _emb = embargo_blocked_fold(
                years, unique_years, test_years, embargo=1, reserve_years=TUNE_VAL_YEARS)
            
            X_train, Y_train = X_seq_np[train_mask], Y_seq_np[train_mask]
            X_val, Y_val = X_seq_np[val_mask], Y_seq_np[val_mask]
            X_test, Y_test = X_seq_np[test_mask], Y_seq_np[test_mask]
            
            Raw_train, Raw_test = Raw_val_da.isel(t=train_mask).values, Raw_val_da.isel(t=test_mask).values
            Truth_train, Truth_test = Y_true_da.isel(t=train_mask).values, Y_true_da.isel(t=test_mask).values
            
            X_mean, X_std = X_train.mean(axis=(0, 1, 2, 3)), X_train.std(axis=(0, 1, 2, 3))
            X_train_norm = np.nan_to_num((X_train - X_mean) / (X_std + 1e-7))
            X_val_norm = np.nan_to_num((X_val - X_mean) / (X_std + 1e-7))
            X_test_norm = np.nan_to_num((X_test - X_mean) / (X_std + 1e-7))
            
            train_steps = len(X_train) // GLOBAL_BATCH_SIZE
            if train_steps == 0:
                print(f"Skipping year {test_years}: Train set too small.")
                continue

            K.clear_session()
            model = tuner.hypermodel.build(best_hps)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
                LossBreakdown(w_tweedie=0.15, w_mae=1.0, w_ssim=15)
            ]
            
            model.fit(
                make_train_dataset(X_train_norm, Y_train),
                validation_data=(X_val_norm, Y_val), 
                epochs=EPOCHS,
                callbacks=callbacks,
                steps_per_epoch=train_steps,
                verbose=0 
            )
            
            # 5. Generate Holdout Predictions (DL Only)
            print("   -> Generating Test Set Predictions (M5 DL)...")
            Y_pred_test = np.maximum(np.squeeze(model.predict(X_test_norm, batch_size=GLOBAL_BATCH_SIZE, verbose=0), -1), 0)
            
            n_test_used = len(Y_pred_test)
            Raw_test_trunc, Truth_test_trunc = Raw_test[:n_test_used], Truth_test[:n_test_used]
            
            # ==================================================================
            # DIAGNOSTIC 1 & 2: PIPELINE AND BIAS CHECKS
            # ==================================================================
            print("\n" + "-"*50)
            print(f"DIAGNOSTICS FOR FOLD: {test_years}")
            
            # Check 1: Global Mean & Max (Is it a pipeline unit offset?)
            t_mean = np.nanmean(Truth_test_trunc)
            p_mean = np.nanmean(Y_pred_test)
            t_max = np.nanmax(Truth_test_trunc)
            p_max = np.nanmax(Y_pred_test)
            
            print(f"  Truth (IMD) Mean: {t_mean:.2f} | Pred (DL) Mean: {p_mean:.2f} (Bias: {p_mean - t_mean:.2f})")
            print(f"  Truth (IMD) Max:  {t_max:.2f} | Pred (DL) Max:  {p_max:.2f}")
            
            # Check 2: Spatial Uniformity (Is it uniform or concentrated?)
            spatial_bias = np.nanmean(Y_pred_test - Truth_test_trunc, axis=0) # Mean bias per pixel over time
            print(f"  Spatial Bias Min: {np.nanmin(spatial_bias):.2f} | Max: {np.nanmax(spatial_bias):.2f} | Median: {np.nanmedian(spatial_bias):.2f}")
            print("-" * 50 + "\n")
            # ==================================================================
            
            # 6. CONVENTIONAL BASELINE (Pure EQM on Raw Model)
            print("   -> Generating Conventional EQM Baseline...")
            Y_eqm_baseline = apply_eqm(Truth_train, Raw_train, Raw_test_trunc)
            
            # 7. VARIANCE RESTORATION (QDM applied directly to M5 DL)
            print("   -> Applying rigorous out-of-sample QDM to M5 predictions...")
            Y_pred_train = np.maximum(np.squeeze(model.predict(X_train_norm, batch_size=GLOBAL_BATCH_SIZE, verbose=0), -1), 0)
            Truth_train_trunc = Truth_train[:len(Y_pred_train)]
            
            Y_dl_qdm_test = apply_qdm_pooled(Truth_train_trunc, Y_pred_train, Y_pred_test)
            
            # 8. Store results
            all_test_times.append(times[test_mask][:n_test_used])
            all_truth_test.append(Truth_test_trunc)
            all_raw_test.append(Raw_test_trunc)
            all_eqm_test.append(Y_eqm_baseline)
            all_dl_test.append(Y_pred_test)
            all_dl_qdm_test.append(Y_dl_qdm_test)

        # ======================================================================
        # PHASE 3: AGGREGATE AND SAVE THE GOLD STANDARD DATASET
        # ======================================================================
        print("\n--- PHASE 3: Aggregating K-Fold Results ---")
        if not all_test_times: continue
            
        final_times = np.concatenate(all_test_times)
        sort_idx = np.argsort(final_times)
        
        lat_vals = Raw_val_da.latitude.values.astype(np.float32)
        lon_vals = Raw_val_da.longitude.values.astype(np.float32)

        ds_out = xr.Dataset({
            'imd_precip':             (('t','latitude','longitude'), np.concatenate(all_truth_test)[sort_idx]),
            'raw_model_precip':       (('t','latitude','longitude'), np.concatenate(all_raw_test)[sort_idx]),
            'eqm_baseline_precip':    (('t','latitude','longitude'), np.concatenate(all_eqm_test)[sort_idx].astype(np.float32)),
            'dl_only_precip':         (('t','latitude','longitude'), np.concatenate(all_dl_test)[sort_idx].astype(np.float32)),
            'm5_dl_qdm_precip':       (('t','latitude','longitude'), np.concatenate(all_dl_qdm_test)[sort_idx].astype(np.float32))
        }, coords={'t': final_times[sort_idx], 'latitude': lat_vals, 'longitude': lon_vals})
        
        for v in ds_out.variables: ds_out[v].encoding = {}
        ds_out['t'].encoding.update({'units': 'days since 1990-01-01', 'calendar': 'standard'})
        
        save_path = os.path.join(VALIDATION_DIR, f"GoldStandard_L3YO_Results_Week{week}{OUT_SUFFIX}.nc")
        if os.path.exists(save_path): os.remove(save_path)
        
        try: ds_out.to_netcdf(save_path, engine='scipy')
        except: ds_out.to_netcdf(save_path, engine='netcdf4')
        print(f"Saved Complete Cross-Validated Dataset: {save_path}")
        
        plot_recent_performance(ds_out, week, num_weeks=8, output_dir=os.path.join(VALIDATION_DIR, "PLOTS"))

        # ======================================================================
        # PHASE 4: TRAIN & SAVE FINAL OPERATIONAL MODEL
        # ======================================================================
        print("\n--- PHASE 4: Training Final Operational Model ---")
        X_final_mean, X_final_std = X_seq_np.mean(axis=(0, 1, 2, 3)), X_seq_np.std(axis=(0, 1, 2, 3))
        X_final_norm = np.nan_to_num((X_seq_np - X_final_mean) / (X_final_std + 1e-7))
        final_steps = len(X_seq_np) // GLOBAL_BATCH_SIZE

        K.clear_session()
        final_model = tuner.hypermodel.build(best_hps)

        final_callbacks = [
            EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4)
        ]

        final_model.fit(
            make_train_dataset(X_final_norm, Y_seq_np),
            epochs=EPOCHS,
            callbacks=final_callbacks,
            steps_per_epoch=final_steps,
            verbose=1
        )

        model_path = os.path.join(MODEL_DIR, f"M5_Final_Week{week}{OUT_SUFFIX}.keras")
        stats_path = os.path.join(MODEL_DIR, f"M5_Final_Week{week}{OUT_SUFFIX}_stats.npz")

        final_model.save(model_path)
        np.savez(stats_path, mean=X_final_mean.astype(np.float32), std=X_final_std.astype(np.float32),
                 channels=np.array(X.channel.values), seq_length=SEQUENCE_LENGTH, week=week)

        print(f"Saved final model: {model_path}")
        print(f"Saved scaler stats: {stats_path}")
        gc.collect()

    print("\nAll weeks processed successfully.")

if __name__ == "__main__":
    main()