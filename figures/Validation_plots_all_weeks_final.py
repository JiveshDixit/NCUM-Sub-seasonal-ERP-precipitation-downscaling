"""

Continuous Skill Score plotting script for NMAE, Spearman correlation, Wasserstein distance
Threshold basedf skill score: FSS
pipeline decomposition script with multiprocessing.
this figures out how much skill each step of the pipeline adds.
i've set it up to use a bunch of cores (like 96) so it runs fast on the hpc.
also loading the p66 thresholds here directly for fss.
"""

import os
import numpy as np
import netCDF4
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import regionmask
from matplotlib.colors import ListedColormap, to_rgb, BoundaryNorm
from scipy.ndimage import uniform_filter
from scipy.stats import wasserstein_distance
from scipy import stats
from functools import partial
import multiprocessing as mp

# basic setup and styling
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# stats config for significance testing
CONFIDENCE_LEVEL = 95  
ALPHA = 1.0 - (CONFIDENCE_LEVEL / 100.0)
N_BOOTSTRAPS = 200     

# pulling in the p66 climatology map for fss instead of a flat threshold
THRESHOLD_FILE = "/home/ncmrwf/bcwc/jivesh/hindcast_erp/DESN/IMD_JJAS_Unified_Thresholds_25km.nc"
FSS_PERCENTILE = "p66" 

# file paths
INDIA_SHP_PATH = "/home/ncmrwf/bcwc/jivesh/Shape_files_India/India_State_Boundary_Updated.shp"
VALIDATION_DIR = "M5_Final_Results_oro_newer"
PLOTS_DIR      = "M5_Final_plots_oro_final"
TARGET_DIR     = os.path.expanduser("~/hindcast_erp/Obs_precip")
TARGET_PATTERN = "IMD_week{week}_sum_25km_on_model_t.nc"
RESULT_PATTERN = "L3YO_Results_Week{week}.nc"
WEEKS = [1, 2, 3, 4]

FONT = {"panel_title": 16, "axis_label": 16, "tick": 14}

# colorbar edges for the bivariate map
CORR_EDGES   = np.array([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
NMAE_EDGES   = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
D_CORR_EDGES = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
D_NMAE_EDGES = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])

# we only care about these four metrics now
STREAMS = ('raw', 'eqm', 'dl', 'dl_qdm')
METRIC_KEYS = ['NMAE', 'Spearman_Corr', 'FSS', 'Wasserstein']
BOOT_METRICS = ['FSS', 'Wasserstein']

STREAM_MAP = {
    'raw': 'raw_model_precip',
    'eqm': 'eqm_baseline_precip',
    'dl': 'dl_only_precip',
    'dl_qdm': 'm5_dl_qdm_precip'
}

# generic data loading and masking helpers
def panel_label(idx): return f"{chr(97 + idx)}."

def save_png_pdf(fig, filename_png):
    fig.savefig(filename_png, bbox_inches='tight', dpi=300)

def get_fill_value(var):
    if hasattr(var, '_FillValue'): return var._FillValue
    if hasattr(var, 'missing_value'): return var.missing_value
    if var.dtype.kind == 'f': return np.nan
    return None

def create_masked_array(data, fill_value):
    if fill_value is not None:
        if np.isnan(fill_value): return np.ma.masked_invalid(data)
        if np.issubdtype(data.dtype, np.floating):
            return np.ma.masked_where(np.isclose(data, fill_value), data)
        return np.ma.masked_equal(data, fill_value)
    return np.ma.masked_array(data, mask=np.zeros_like(data, dtype=bool))

def force_shape_to_time_lat_lon(arr, lat_size, lon_size):
    arr = arr.squeeze()
    if arr.shape[-2] != lat_size or arr.shape[-1] != lon_size:
        if arr.shape[-2] == lon_size and arr.shape[-1] == lat_size:
            arr = arr.transpose(0, 2, 1)
    return arr.reshape(-1, lat_size, lon_size)

def generate_static_mask(lons, lats, shp_path):
    # doing this once to save time so we don't recalculate the shapefile every loop
    if not os.path.exists(shp_path):
        return np.zeros((len(lats), len(lons)), dtype=bool)
    try:
        gdf = gpd.read_file(shp_path)
        mask = regionmask.Regions(gdf.geometry).mask(lons, lats)
        return np.isnan(mask.values)
    except Exception:
        return np.zeros((len(lats), len(lons)), dtype=bool)

def apply_static_mask(data_ma, shp_mask):
    if data_ma.ndim == 3:
        combined = np.ma.getmaskarray(data_ma) | shp_mask[np.newaxis, :, :]
    else:
        combined = np.ma.getmaskarray(data_ma) | shp_mask
        
    raw_data = np.asarray(data_ma, dtype=float)
    raw_data[combined] = np.nan
    return np.ma.masked_invalid(raw_data)

def _load_and_mask_var(ds, var_name, lats, lons, shp_mask):
    raw_arr = ds.variables[var_name][:]
    arr = create_masked_array(raw_arr, get_fill_value(ds.variables[var_name]))
    arr = force_shape_to_time_lat_lon(arr, len(lats), len(lons))
    return apply_static_mask(arr, shp_mask)

def common_mask(a, b):
    return np.ma.getmaskarray(a) | np.ma.getmaskarray(b) | np.isnan(a) | np.isnan(b)

# stats and metric calculations
def make_seasonal_block_indices(season_ids, n_boot=1000, seed=0):
    season_ids = np.asarray(season_ids)
    seasons = np.unique(season_ids)
    blocks = [np.where(season_ids == s)[0] for s in seasons]
    block_arr = np.stack(blocks)
    n_seasons = block_arr.shape[0]
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_seasons, size=(n_boot, n_seasons))
    return block_arr[draw].reshape(n_boot, -1)

def pval_mean_diff(ts_err1, ts_err2):
    if hasattr(ts_err1, 'filled'): ts_err1 = ts_err1.filled(np.nan)
    if hasattr(ts_err2, 'filled'): ts_err2 = ts_err2.filled(np.nan)
    diff = np.abs(ts_err1) - np.abs(ts_err2)
    mean = np.nanmean(diff, axis=0)
    std = np.nanstd(diff, axis=0, ddof=1)
    n = np.sum(~np.isnan(diff), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = mean / (std / np.sqrt(n) + 1e-8)
    return 2.0 * stats.t.sf(np.abs(t_stat), df=n - 1)

def pval_mean(ts, popmean=0.0):
    if hasattr(ts, 'filled'): ts = ts.filled(np.nan)
    mean = np.nanmean(ts, axis=0)
    std = np.nanstd(ts, axis=0, ddof=1)
    n = np.sum(~np.isnan(ts), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (mean - popmean) / (std / np.sqrt(n) + 1e-8)
    return 2.0 * stats.t.sf(np.abs(t_stat), df=n - 1)

def pval_corr(r, n):
    if hasattr(r, 'filled'): r = r.filled(np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = r * np.sqrt(n - 2) / (np.sqrt(1 - r**2) + 1e-8)
    return 2.0 * stats.t.sf(np.abs(t_stat), df=n - 2)

def pval_corr_diff(r1, r2, n):
    if hasattr(r1, 'filled'): r1 = r1.filled(np.nan)
    if hasattr(r2, 'filled'): r2 = r2.filled(np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        z1 = 0.5 * np.log((1 + r1) / (1 - r1 + 1e-8))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2 + 1e-8))
        z_stat = (z1 - z2) / np.sqrt(2.0 / (n - 3) + 1e-8)
    return 2.0 * stats.norm.sf(np.abs(z_stat))

def fdr_field_mask(pvals, alpha_fdr=0.10):
    # checking for false discovery rate
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    ok = np.isfinite(flat)
    pv = flat[ok]
    m = pv.size
    out = np.zeros(flat.size, dtype=bool)
    if m == 0: return out.reshape(p.shape)
    ranked = np.sort(pv)
    bh_thresh = alpha_fdr * (np.arange(1, m + 1) / m)
    passed = ranked <= bh_thresh
    crit = ranked[passed][-1] if passed.any() else -1.0
    out[ok] = pv <= crit
    return out.reshape(p.shape)

def calculate_temporal_correlation_map(data1_ma, data2_ma):
    # simple spearman rank correlation
    da1 = xr.DataArray(data1_ma, dims=['time', 'lat', 'lon'])
    da2 = xr.DataArray(data2_ma, dims=['time', 'lat', 'lon'])
    mask = np.ma.getmaskarray(data1_ma)
    da1 = da1.where(~mask)
    da2 = da2.where(~mask)
    
    corr_map = xr.corr(da1.rank(dim='time'), da2.rank(dim='time'), dim='time')
    result = corr_map.values
    m = np.ma.getmaskarray(data1_ma[0]) | np.isnan(result)
    return np.ma.masked_array(result, mask=m)

def calculate_fss_map(model_ma, obs_ma, threshold, window=5):
    # fractions skill score, threshold is now a 2D map
    t, ny, nx = model_ma.shape
    common_m = np.ma.getmaskarray(model_ma) | np.ma.getmaskarray(obs_ma)
    valid_m = (~common_m).astype(float) 
    
    mod_data = np.ma.filled(model_ma, 0.0)
    obs_data = np.ma.filled(obs_ma, 0.0)
    
    # numpy automatically broadcasts the 2D p66 map across the time dimension
    mod_bin = ((mod_data >= threshold) & (valid_m == 1.0)).astype(float)
    obs_bin = ((obs_data >= threshold) & (valid_m == 1.0)).astype(float)
    
    mse_sum = np.zeros((ny, nx))
    denom_sum = np.zeros((ny, nx))
    
    for tt in range(t):
        m_filt = uniform_filter(mod_bin[tt], size=window, mode='constant', cval=0.0)
        o_filt = uniform_filter(obs_bin[tt], size=window, mode='constant', cval=0.0)
        v_filt = uniform_filter(valid_m[tt], size=window, mode='constant', cval=0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mod_frac = np.where(v_filt > 0, m_filt / v_filt, 0.0)
            obs_frac = np.where(v_filt > 0, o_filt / v_filt, 0.0)
        mse_sum += (obs_frac - mod_frac) ** 2
        denom_sum += (obs_frac ** 2 + mod_frac ** 2)
        
    with np.errstate(divide='ignore', invalid='ignore'):
        fss_map = 1.0 - (mse_sum / denom_sum)
    fss_map[denom_sum == 0] = np.nan
    fss_map[common_m[0]] = np.nan
    return np.ma.masked_array(fss_map, mask=common_m[0] | np.isnan(fss_map))

def calculate_wasserstein_map(model_ma, obs_ma):
    t, ny, nx = model_ma.shape
    wd = np.full((ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            m = model_ma[:, i, j]; o = obs_ma[:, i, j]
            mask = np.ma.getmaskarray(m) | np.ma.getmaskarray(o)
            m = np.asarray(m)[~mask]; o = np.asarray(o)[~mask]
            if len(m) < 10: continue
            wd[i, j] = wasserstein_distance(m, o)
    return np.ma.masked_invalid(wd)

# parallel worker tasks
def worker_fast_metrics(args):
    """gets the simple base metrics and false discovery rate stats."""
    w, lats, lons, shp_mask, p66_map = args
    print(f"  [FAST] Processing base metrics for Week {w}...")
    
    fname = os.path.join(VALIDATION_DIR, RESULT_PATTERN.format(week=w))
    S_w = {m: {s: None for s in STREAMS} for m in METRIC_KEYS}
    SIG_w = {m: {s: None for s in STREAMS} for m in METRIC_KEYS}
    
    with netCDF4.Dataset(fname, 'r') as ds:
        imd = _load_and_mask_var(ds, 'imd_precip', lats, lons, shp_mask)
        fields = {s: _load_and_mask_var(ds, STREAM_MAP[s], lats, lons, shp_mask) for s in STREAMS}

    N = imd.shape[0]
    e = {s: fields[s] - imd for s in STREAMS}
    imd_mean = imd.mean(0)
    # prevent zero division for nmae
    safe = np.ma.masked_where(imd_mean < 0.1, imd_mean)

    for s in STREAMS:
        S_w['NMAE'][s] = np.ma.abs(e[s]).mean(0) / safe
        S_w['Spearman_Corr'][s] = calculate_temporal_correlation_map(fields[s], imd)
        S_w['FSS'][s] = calculate_fss_map(fields[s], imd, threshold=p66_map, window=5)
        S_w['Wasserstein'][s] = calculate_wasserstein_map(fields[s], imd)

    SIG_w['NMAE']['raw'] = fdr_field_mask(pval_mean(np.ma.abs(e['raw']), 0.0), alpha_fdr=ALPHA)
    SIG_w['Spearman_Corr']['raw'] = fdr_field_mask(pval_corr(S_w['Spearman_Corr']['raw'], N), alpha_fdr=ALPHA)

    for s in ('eqm', 'dl', 'dl_qdm'):
        SIG_w['NMAE'][s] = fdr_field_mask(pval_mean_diff(e[s], e['raw']), alpha_fdr=ALPHA)
        SIG_w['Spearman_Corr'][s] = fdr_field_mask(pval_corr_diff(S_w['Spearman_Corr'][s], S_w['Spearman_Corr']['raw'], N), alpha_fdr=ALPHA)

    return w, S_w, SIG_w

def worker_bootstrap_chunk(args):
    """small micro-chunk for the bootstrapping so we can max out the cores."""
    w, stream, metric_name, n_boot, seed, lats, lons, shp_mask, p66_map = args
    fname = os.path.join(VALIDATION_DIR, RESULT_PATTERN.format(week=w))
    
    with xr.open_dataset(fname) as ds_xr:
        season_ids = ds_xr['t'].dt.year.values
        
    with netCDF4.Dataset(fname, 'r') as ds:
        imd = _load_and_mask_var(ds, 'imd_precip', lats, lons, shp_mask)
        raw = _load_and_mask_var(ds, 'raw_model_precip', lats, lons, shp_mask)
        mod = _load_and_mask_var(ds, STREAM_MAP[stream], lats, lons, shp_mask)

    if metric_name == 'FSS': 
        metric_fn = partial(calculate_fss_map, threshold=p66_map, window=5)
    elif metric_name == 'Wasserstein': 
        metric_fn = calculate_wasserstein_map

    idx_mat = make_seasonal_block_indices(season_ids, n_boot=n_boot, seed=seed)
    diffs = np.empty((n_boot,) + imd.shape[1:], dtype=float)

    for b in range(n_boot):
        ii = idx_mat[b]
        m_eval = metric_fn(mod[ii], imd[ii])
        r_eval = metric_fn(raw[ii], imd[ii])

        if hasattr(m_eval, 'filled'): m_eval = m_eval.filled(np.nan)
        if hasattr(r_eval, 'filled'): r_eval = r_eval.filled(np.nan)
        diffs[b] = m_eval - r_eval

    return w, stream, metric_name, diffs

# map styling and plotting tools
def diff_config(metric_name):
    if metric_name == 'Spearman_Corr':
        return ('seismic_r', -1.0, 1.0, None, 'unitless', 'seismic_r', -0.6, 0.6, r"$\Delta$ Skill (Blue $\rightarrow$ better)", +1, 'neither')
    if metric_name == 'FSS':
        return ('viridis', 0.0, 1.0, None, 'unitless', 'seismic_r', -0.3, 0.3, r"$\Delta$ FSS (Blue $\rightarrow$ better)", +1, 'neither')
    if metric_name == 'Wasserstein':
        return ('plasma_r', 0.0, 50.0, None, 'mm', 'seismic_r', -30.0, 30.0, r"$\Delta$ Dist (Blue $\rightarrow$ reduced)", -1, 'max')
    if metric_name == 'NMAE':
        return ('plasma_r', 0.0, 2.0, None, 'unitless', 'seismic_r', -1.0, 1.0, r"$\Delta$ NMAE (Blue $\rightarrow$ reduced)", -1, 'max')
    return ('plasma_r', 0.0, 100.0, None, 'mm', 'seismic_r', -50, 50, r"$\Delta$ Metric (Blue $\rightarrow$ reduced)", -1, 'max')

def _signed_for_plot(model_map, raw_map, improvement_sign):
    if improvement_sign > 0: return model_map - raw_map
    return raw_map - model_map

def add_stippling(ax, lons, lats, sig_mask):
    if sig_mask is not None:
        lon_g, lat_g = np.meshgrid(lons, lats)
        ax.scatter(lon_g[sig_mask], lat_g[sig_mask], s=0.4, color='black', alpha=0.5, transform=ccrs.PlateCarree(), marker='.', zorder=3)
        
def beautify_map(ax, title, shape_geoms=None):
    ax.set_extent([66.0, 100.0, 6.0, 38.0], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    if shape_geoms: ax.add_geometries(shape_geoms, ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.6)
    else: ax.coastlines(linewidth=0.6)
    if title: ax.set_title(title, fontsize=FONT["panel_title"], fontweight='bold', pad=30)

def beautify_panel_map(ax, title, shape_geoms=None):
    ax.set_extent([66.0, 100.0, 6.0, 38.0], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    if shape_geoms: ax.add_geometries(shape_geoms, ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5)
    else: ax.coastlines(linewidth=0.5)
    if title: ax.set_title(title, fontsize=FONT["panel_title"], fontweight='bold', pad=25)

def get_bivariate_cmap(n=6):
    c_ll = np.array(to_rgb("#f2f2f2")); c_hl = np.array(to_rgb("#00e5ff"))
    c_lh = np.array(to_rgb("#ff007f")); c_hh = np.array(to_rgb("#6600cc"))
    colors = []
    for y in np.linspace(0, 1, n):
        for x in np.linspace(0, 1, n):
            colors.append((1 - x) * (1 - y) * c_ll + x * (1 - y) * c_hl + (1 - x) * y * c_lh + x * y * c_hh)
    return ListedColormap(colors)

def get_bivariate_diverging_cmap(n=4):
    c_blue = np.array(to_rgb("#0055ff")); c_orange = np.array(to_rgb("#ff9900"))
    c_purple = np.array(to_rgb("#aa00ff")); c_red = np.array(to_rgb("#ff0000"))
    colors = []
    for y_idx in range(n):
        for x_idx in range(n):
            x = x_idx / (n - 1); y = y_idx / (n - 1)
            color = ((1 - x) * (1 - y) * c_blue + x * (1 - y) * c_orange + (1 - x) * y * c_purple + x * y * c_red)
            ix = (x_idx in (1, 2)); iy = (y_idx in (1, 2))
            if ix and iy: color = color * 0.4 + np.array([1, 1, 1]) * 0.6
            elif ix or iy: color = color * 0.7 + np.array([1, 1, 1]) * 0.3
            colors.append(color)
    return ListedColormap(colors)

def classify_independent(x, y, x_edges, y_edges):
    mask = common_mask(x, y)
    x_m = np.ma.masked_array(x, mask=mask); y_m = np.ma.masked_array(y, mask=mask)
    xi = np.digitize(x_m, x_edges) - 1; yi = np.digitize(y_m, y_edges) - 1
    n = len(x_edges) - 1
    xi = np.clip(xi, 0, n - 1); yi = np.clip(yi, 0, n - 1)
    return np.ma.masked_array(yi * n + xi, mask=mask)

def get_smart_labels(edges):
    for decimals in range(1, 7):
        labels = [f"{v:.{decimals}f}" for v in edges]
        if len(set(labels)) < len(labels): continue
        if len([l for l in labels if float(l) == 0]) > 1: continue
        return [l.replace("-", "") if (float(l) == 0 and l.startswith("-")) else l for l in labels]
    return [f"{v:.1e}" for v in edges]

def setup_legend_ticks(ax, x_edges, y_edges, n, xlabel, ylabel):
    ax.set_xticks(np.arange(n + 1)); ax.set_yticks(np.arange(n + 1))
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels(get_smart_labels(x_edges), rotation=45, ha='right', fontsize=FONT["tick"], fontweight='bold')
    ax.set_yticklabels(get_smart_labels(y_edges), fontsize=FONT["tick"], fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=FONT["axis_label"], fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=FONT["axis_label"], fontweight='bold')
    for s in ax.spines.values(): s.set_visible(False)

def plot_skill_decomposition(metric_name, raw_l, eqm_l, dl_l, dl_qdm_l, sig_raw_l, sig_eqm_l, sig_dl_l, sig_dl_qdm_l, lons, lats, shape_geoms, filename):
    print(f"Plotting Univariate: {filename} ({metric_name})...")
    (cmap_abs, vmin_a, vmax_a, norm_a, units, cmap_d, vmin_d, vmax_d, dlabel, imp_sign, ext_abs) = diff_config(metric_name)
    lon_g, lat_g = np.meshgrid(lons, lats)
    titles = [f"Raw NCUM {metric_name}", r"$\Delta$ (EQM $-$ Raw)", r"$\Delta$ (M5 DL $-$ Raw)", r"$\Delta$ (M5 DL+QDM $-$ Raw)"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    mesh_abs = mesh_d = None; panel = 0
    for i, w in enumerate(WEEKS):
        raw = raw_l[i]
        cols = [
            (raw_l[i], cmap_abs, norm_a, vmin_a, vmax_a, sig_raw_l[i], True),
            (_signed_for_plot(eqm_l[i],    raw, imp_sign), cmap_d, None, vmin_d, vmax_d, sig_eqm_l[i],    False),
            (_signed_for_plot(dl_l[i],     raw, imp_sign), cmap_d, None, vmin_d, vmax_d, sig_dl_l[i],     False),
            (_signed_for_plot(dl_qdm_l[i], raw, imp_sign), cmap_d, None, vmin_d, vmax_d, sig_dl_qdm_l[i], False),
        ]
        axes[i, 0].text(-0.10, 0.5, f"Week {w}", transform=axes[i, 0].transAxes, rotation=90, va='center', ha='right', fontsize=18, fontweight='bold')
        for j, (data, cmap, norm, vm, vx, sig, is_abs) in enumerate(cols):
            ax = axes[i, j]
            ax.text(0.02, 1.1, panel_label(panel), transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left'); panel += 1
            kw = dict(transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')
            if norm is not None: kw['norm'] = norm
            else: kw['vmin'], kw['vmax'] = vm, vx
            mesh = ax.pcolormesh(lon_g, lat_g, data, **kw)
            add_stippling(ax, lons, lats, sig)
            beautify_map(ax, titles[j] if i == 0 else "", shape_geoms)
            if is_abs: mesh_abs = mesh
            else: mesh_d = mesh

    cax1 = fig.add_axes([0.15, 0.08, 0.15, 0.015])
    cb1 = fig.colorbar(mesh_abs, cax=cax1, orientation='horizontal', extend=ext_abs)
    cb1.set_label(f"{metric_name} ({units})", fontweight='bold', fontsize=14)
    cax2 = fig.add_axes([0.35, 0.08, 0.50, 0.015])
    cb2 = fig.colorbar(mesh_d, cax=cax2, orientation='horizontal', extend='both')
    cb2.set_label(dlabel, fontweight='bold', fontsize=14)

    plt.subplots_adjust(bottom=0.12, top=0.90, hspace=0.02, wspace=0.01)
    save_png_pdf(plt.gcf(), filename); plt.close()

def plot_bivariate_decomposition(m1_raw, m2_raw, m1_eqm, m2_eqm, m1_dl, m2_dl, m1_dl_qdm, m2_dl_qdm, name1, name2, main_edges, diff_edges, sig_eqm, sig_dl, sig_dl_qdm, lons, lats, shape_geoms, filename):
    print(f"Plotting Bivariate: {filename} ({name1} x {name2})...")
    x_edges, y_edges = main_edges
    dx_edges, dy_edges = diff_edges
    n_main = len(x_edges) - 1; n_diff = len(dx_edges) - 1
    cmap_main = get_bivariate_cmap(n_main)
    cmap_diff = get_bivariate_diverging_cmap(n_diff)
    norm_main = BoundaryNorm(np.arange(n_main * n_main + 1), ncolors=n_main * n_main)
    norm_diff = BoundaryNorm(np.arange(n_diff * n_diff + 1), ncolors=n_diff * n_diff)
    lon_g, lat_g = np.meshgrid(lons, lats)
    titles = ["Raw NCUM", r"$\Delta$ (EQM $-$ Raw)", r"$\Delta$ (M5 DL $-$ Raw)", r"$\Delta$ (M5 DL+QDM $-$ Raw)"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 22), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    panel = 0
    for i in range(4):
        cls_raw    = classify_independent(m1_raw[i],    m2_raw[i],    x_edges, y_edges)
        cls_eqm    = classify_independent(m1_eqm[i] - m1_raw[i], m2_eqm[i] - m2_raw[i], dx_edges, dy_edges)
        cls_dl     = classify_independent(m1_dl[i] - m1_raw[i], m2_dl[i] - m2_raw[i], dx_edges, dy_edges)
        cls_dl_qdm = classify_independent(m1_dl_qdm[i] - m1_raw[i], m2_dl_qdm[i] - m2_raw[i], dx_edges, dy_edges)

        cols = [
            (cls_raw,    cmap_main, norm_main, None,         True),
            (cls_eqm,    cmap_diff, norm_diff, sig_eqm[i],    False),
            (cls_dl,     cmap_diff, norm_diff, sig_dl[i],     False),
            (cls_dl_qdm, cmap_diff, norm_diff, sig_dl_qdm[i], False),
        ]
        axes[i, 0].text(-0.10, 0.5, f"Week {i+1}", transform=axes[i, 0].transAxes, rotation=90, va='center', ha='right', fontsize=20, fontweight='bold')
        for j, (data, cmap, norm, sig, is_abs) in enumerate(cols):
            ax = axes[i, j]
            ax.text(0.02, 1.1, panel_label(panel), transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left'); panel += 1
            ax.pcolormesh(lon_g, lat_g, data, cmap=cmap, norm=norm, shading='auto')
            if not is_abs: add_stippling(ax, lons, lats, sig)
            beautify_panel_map(ax, titles[j] if i == 0 else "", shape_geoms)

    leg1 = fig.add_axes([0.20, 0.10, 0.15, 0.13])
    leg1.imshow(np.arange(n_main**2).reshape(n_main, n_main), cmap=cmap_main, origin='lower', extent=[0, n_main, 0, n_main])
    setup_legend_ticks(leg1, x_edges, y_edges, n_main, name1, name2)
    leg1.set_title("Absolute (Raw)", fontsize=13, fontweight='bold')

    leg2 = fig.add_axes([0.60, 0.10, 0.15, 0.13])
    leg2.imshow(np.arange(n_diff**2).reshape(n_diff, n_diff), cmap=cmap_diff, origin='lower', extent=[0, n_diff, 0, n_diff])
    setup_legend_ticks(leg2, dx_edges, dy_edges, n_diff, f"$\\Delta${name1}", f"$\\Delta${name2}")
    leg2.vlines(np.arange(1, n_diff), 0, n_diff, colors='white', linestyles='--', linewidth=0.8)
    leg2.hlines(np.arange(1, n_diff), 0, n_diff, colors='white', linestyles='--', linewidth=0.8)
    leg2.set_title(r"$\Delta$ vs Raw", fontsize=13, fontweight='bold')

    plt.subplots_adjust(bottom=0.26, top=0.90, hspace=0.01, wspace=0.01)
    save_png_pdf(plt.gcf(), filename); plt.close()

# main driver code
def run():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    geoms = (list(shpreader.Reader(INDIA_SHP_PATH).geometries()) if os.path.exists(INDIA_SHP_PATH) else None)

    # 1. grab lats/lons and calculate the static mask once
    sample_file = os.path.join(VALIDATION_DIR, RESULT_PATTERN.format(week=1))
    with netCDF4.Dataset(sample_file, 'r') as ds:
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]
    
    print("Pre-computing spatial shapefile mask (doing this once to save CPU time)...")
    STATIC_SHP_MASK = generate_static_mask(lons, lats, INDIA_SHP_PATH)

    print(f"Pre-computing 2D '{FSS_PERCENTILE}' Threshold map for FSS...")
    with xr.open_dataset(THRESHOLD_FILE) as ds_thr:
        # cleaning up lat/lon naming so it aligns with the grid
        renames = {}
        if 'lat' in ds_thr.coords: renames['lat'] = 'latitude'
        if 'lon' in ds_thr.coords: renames['lon'] = 'longitude'
        if renames: ds_thr = ds_thr.rename(renames)
        # extracting the 2d map
        P66_MAP = ds_thr[FSS_PERCENTILE].sel(latitude=lats, longitude=lons, method="nearest").values

    S = {m: {s: [] for s in STREAMS} for m in METRIC_KEYS}
    SIG = {m: {s: [] for s in STREAMS} for m in METRIC_KEYS}

    n_workers = min(96, mp.cpu_count()) 
    print(f"\n🚀 Booting Multiprocessing Pool with {n_workers} active cores...")
    
    with mp.Pool(n_workers) as pool:
        
        # phase 1: fast base metrics and fdr (sequential loop over 4 weeks)
        fast_tasks = [(w, lats, lons, STATIC_SHP_MASK, P66_MAP) for w in WEEKS]
        fast_results = pool.map(worker_fast_metrics, fast_tasks)
        
        for res in fast_results:
            w_res, S_w, SIG_w = res
            for m in METRIC_KEYS:
                for s in STREAMS:
                    S[m][s].append(S_w[m][s])
                    SIG[m][s].append(SIG_w[m][s])

        # phase 2: spreading bootstrap chunks to saturate the hpc
        print("\nSpawning flattened bootstrap tasks to saturate all cores...")
        chunks_per_task = 4  
        boots_per_chunk = N_BOOTSTRAPS // chunks_per_task  
        
        boot_tasks = []
        for w in WEEKS:
            for stream in ('eqm', 'dl', 'dl_qdm'):
                for metric in BOOT_METRICS:
                    for c in range(chunks_per_task):
                        seed = hash(f"{w}_{stream}_{metric}_{c}") % (2**32)
                        # pass p66_map right into the arguments so workers don't read it again
                        boot_tasks.append((w, stream, metric, boots_per_chunk, seed, lats, lons, STATIC_SHP_MASK, P66_MAP))
        
        print(f"Generated {len(boot_tasks)} micro-tasks. Distributing...")
        boot_results = pool.map(worker_bootstrap_chunk, boot_tasks)

    # phase 3: piece it together and calculate significance
    print("\nAggregating and resolving statistical significance thresholds...")
    boot_diffs = {w: {s: {m: [] for m in BOOT_METRICS} for s in ('eqm', 'dl', 'dl_qdm')} for w in WEEKS}
    
    for res in boot_results:
        w_res, stream, m, diffs = res
        boot_diffs[w_res][stream][m].append(diffs)
        
    for w_idx, w in enumerate(WEEKS):
        for s in ('eqm', 'dl', 'dl_qdm'):
            for m in BOOT_METRICS:
                all_diffs = np.concatenate(boot_diffs[w][s][m], axis=0) 
                
                higher_is_better = True if m not in ['Wasserstein'] else False
                
                with np.errstate(invalid='ignore'):
                    lo = np.nanpercentile(all_diffs, 100 * (ALPHA / 2.0), axis=0)
                    hi = np.nanpercentile(all_diffs, 100 * (1.0 - ALPHA / 2.0), axis=0)

                sig_mask = (lo > 0) if higher_is_better else (hi < 0)
                SIG[m][s][w_idx] = sig_mask

    # plotting out the final maps
    print("\nStarting plotting sequence...")
    for metric in METRIC_KEYS:
        plot_skill_decomposition(
            metric,
            S[metric]['raw'], S[metric]['eqm'], S[metric]['dl'], S[metric]['dl_qdm'],
            SIG[metric]['raw'], SIG[metric]['eqm'], SIG[metric]['dl'], SIG[metric]['dl_qdm'],
            lons, lats, geoms,
            os.path.join(PLOTS_DIR, f"Pipeline_Decomposition_{metric}.png")
        )

    plot_bivariate_decomposition(
        S['Spearman_Corr']['raw'], S['NMAE']['raw'], S['Spearman_Corr']['eqm'], S['NMAE']['eqm'],
        S['Spearman_Corr']['dl'], S['NMAE']['dl'], S['Spearman_Corr']['dl_qdm'], S['NMAE']['dl_qdm'],
        "Correlation", "NMAE", (CORR_EDGES, NMAE_EDGES), (D_CORR_EDGES, D_NMAE_EDGES),
        SIG['Spearman_Corr']['eqm'], SIG['Spearman_Corr']['dl'], SIG['Spearman_Corr']['dl_qdm'],
        lons, lats, geoms,
        os.path.join(PLOTS_DIR, "Pipeline_Bivariate_Corr_NMAE.png"))


if __name__ == "__main__":
    run()