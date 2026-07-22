"""
regional threshold metrics script (pod, far, ets, hss) with multiprocessing.
this calculates the categorical scores and runs a moving-block bootstrap
to see if the improvements are statistically significant.

this version generates a 'mega performance portrait' heatmap.
it plots all models (eqm, dl, dl+qdm) as rows and metrics as columns.
inside each panel, the x-axis expands to fit all three thresholds (p33, p50, p66)
side-by-side, giving a complete bird's-eye view of everything in one image.
"""

import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import geopandas as gpd
import regionmask
import os
import matplotlib as mpl
import multiprocessing as mp
import scipy.stats as st

# basic configuration and styling
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# file paths
DATA_DIR = "M5_Final_Results_oro_newer"
PLOTS_DIR = "M5_Final_plots_oro_final"
THRESHOLD_FILE = "/home/ncmrwf/bcwc/jivesh/hindcast_erp/DESN/IMD_JJAS_Unified_Thresholds_25km.nc"
SHAPE_DIR = "/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/Homogenous_rainfal_India/"

# plot settings
THRESHOLDS = ["p33", "p50", "p66"]
REGIONS = [
    "Hilly Regions", "Northwest", "Central Northeast", 
    "Northeast", "West Central", "Peninsular"
]
METRICS_TO_PLOT = ["POD", "FAR", "ETS", "HSS"]
METRIC_INDICES = {"POD": 0, "FAR": 1, "ETS": 2, "HSS": 3}

# statistics setup
CONFIDENCE_PERCENT = 95  
Z_SCORE = st.norm.ppf(1 - (1 - CONFIDENCE_PERCENT / 100.0) / 2.0)
N_BOOTSTRAP = 1000       

# fonts and ticks
TICK_SIZE = 14           
TICK_WEIGHT = 'bold'     
LABEL_SIZE = 16          
TITLE_SIZE = 20          

plt.rcParams['font.family'] = 'sans-serif'
bold_font = FontProperties(weight='bold', size=14)
title_font = FontProperties(weight='bold', size=TITLE_SIZE)
region_label_font = FontProperties(weight='bold', size=LABEL_SIZE)

# ================= 1. FAST DATA EXTRACTION =================

def build_region_masks(lats, lons):
    files = {
        "Northwest": "Northwest.shp",
        "Central Northeast": "Central_Northeast.shp",
        "Northeast": "Northeast.shp",
        "West Central": "West_Central.shp",
        "Peninsular": "South_Peninsular.shp",
        "Hilly Regions": "Hilly_Regions.shp",
    }
    lon2d, lat2d = np.meshgrid(lons, lats)
    masks = {}
    print("Pre-computing spatial shapefile masks...")
    for name, shp in files.items():
        shp_path = os.path.join(SHAPE_DIR, shp)
        if os.path.exists(shp_path):
            gdf = gpd.read_file(shp_path)
            reg = regionmask.Regions(gdf.geometry)
            mask = reg.mask(lon2d, lat2d)
            masks[name] = ~np.isnan(mask.values)
    return masks

def contingency_vectorized(obs, pred, thr):
    obs_event  = obs >= thr
    pred_event = pred >= thr
    H = np.sum(obs_event & pred_event, axis=1)
    F = np.sum(~obs_event & pred_event, axis=1)
    M = np.sum(obs_event & ~pred_event, axis=1)
    C = np.sum(~obs_event & ~pred_event, axis=1)
    eps = 1e-8
    N = H + F + M + C
    POD = H / (H + M + eps)
    FAR = F / (H + F + eps)
    H_rand = ((H + F) * (H + M)) / (N + eps)
    ETS = (H - H_rand) / (H + F + M - H_rand + eps)
    hss_num = 2 * (H * C - F * M)
    hss_den = (H + M)*(M + C) + (H + F)*(F + C)
    HSS = hss_num / (hss_den + eps)
    return np.stack([POD, FAR, ETS, HSS], axis=1)

def contingency_1d(obs, pred, thr):
    obs_event  = obs >= thr
    pred_event = pred >= thr
    H = np.sum(obs_event & pred_event)
    F = np.sum(~obs_event & pred_event)
    M = np.sum(obs_event & ~pred_event)
    C = np.sum(~obs_event & ~pred_event)
    eps = 1e-8
    N = H + F + M + C
    POD = H / (H + M + eps)
    FAR = F / (H + F + eps)
    H_rand = ((H + F) * (H + M)) / (N + eps)
    ETS = (H - H_rand) / (H + F + M - H_rand + eps)
    hss_num = 2 * (H * C - F * M)
    hss_den = (H + M)*(M + C) + (H + F)*(F + C)
    HSS = hss_num / (hss_den + eps)
    return POD, FAR, ETS, HSS

def make_seasonal_block_indices(season_ids, n_boot=1000, seed=0):
    season_ids = np.asarray(season_ids)
    seasons = np.unique(season_ids)
    blocks = [np.where(season_ids == s)[0] for s in seasons]
    lens = {len(b) for b in blocks}
    if len(lens) != 1:
        min_len = min(lens)
        blocks = [b[:min_len] for b in blocks]
    block_arr = np.stack(blocks)                       
    n_seasons = block_arr.shape[0]
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_seasons, size=(n_boot, n_seasons))
    return block_arr[draw].reshape(n_boot, -1)         

def worker_extract_base_data(args):
    week, masks, lats, lons = args
    file_path = os.path.join(DATA_DIR, f"L3YO_Results_Week{week}.nc")
    if not os.path.exists(file_path): return None

    with xr.open_dataset(file_path) as ds_xr: season_ids = ds_xr['t'].dt.year.values
    
    with netCDF4.Dataset(file_path) as ds:
        obs = ds.variables["imd_precip"][:]
        raw = ds.variables["raw_model_precip"][:]
        eqm_ts = ds.variables["eqm_baseline_precip"][:] 
        dl_ts = ds.variables["dl_only_precip"][:]
        dl_qdm_ts = ds.variables["m5_dl_qdm_precip"][:]

    ds_thr = xr.open_dataset(THRESHOLD_FILE)
    if "lat" in ds_thr.coords: ds_thr = ds_thr.rename({"lat": "latitude"})
    if "lon" in ds_thr.coords: ds_thr = ds_thr.rename({"lon": "longitude"})
    thr_ds = ds_thr.sel(latitude=lats, longitude=lons, method="nearest")

    def get_ts(data_3d, mask_2d): return np.nanmean(np.where(mask_2d, data_3d, np.nan), axis=(1,2))

    base_scores = {}
    timeseries_data = {}
    for tvar in THRESHOLDS:
        thr_map = thr_ds[tvar].values
        base_scores[tvar] = {}
        timeseries_data[tvar] = {}
        for rname in REGIONS:
            if rname not in masks: continue
            mask = masks[rname]
            thr_region = np.nanmean(thr_map[mask])
            
            obs_reg, raw_reg = get_ts(obs, mask), get_ts(raw, mask)
            eqm_reg, dl_reg, dl_qdm_reg = get_ts(eqm_ts, mask), get_ts(dl_ts, mask), get_ts(dl_qdm_ts, mask)

            base_scores[tvar][rname] = {
                "RAW": contingency_1d(obs_reg, raw_reg, thr_region),
                "EQM": contingency_1d(obs_reg, eqm_reg, thr_region),
                "DL": contingency_1d(obs_reg, dl_reg, thr_region),
                "M5_DL_QDM": contingency_1d(obs_reg, dl_qdm_reg, thr_region)
            }
            timeseries_data[tvar][rname] = {
                "thr": thr_region, "obs": obs_reg, "raw": raw_reg, 
                "eqm": eqm_reg, "dl": dl_reg, "dl_qdm": dl_qdm_reg
            }
    ds_thr.close()
    return week, season_ids, base_scores, timeseries_data

def worker_bootstrap_1d(args):
    week, tvar, rname, mod_key, sig_key, obs_ts, raw_ts, mod_ts, thr_val, season_ids, seed = args
    if len(obs_ts) < 2: return week, tvar, rname, sig_key, np.zeros(4)

    idx = make_seasonal_block_indices(season_ids, n_boot=N_BOOTSTRAP, seed=seed)
    scores_raw = contingency_vectorized(obs_ts[idx], raw_ts[idx], thr_val)
    scores_mod = contingency_vectorized(obs_ts[idx], mod_ts[idx], thr_val)
    
    diffs = scores_mod - scores_raw
    sig_result = Z_SCORE * np.std(diffs, axis=0)
    return week, tvar, rname, sig_key, sig_result


# ================= 2. CREATIVE PLOTTING =================

def plot_mega_heatmap(weekly_scores, save_name="Mega_Performance_Portrait"):
    """
    THE MEGA PERFORMANCE PORTRAIT
    fits all models, all metrics, all regions, all leads, AND all thresholds 
    into one super-dense graphic.
    """
    models = ["EQM", "DL", "M5_DL_QDM"]
    model_titles = ["EQM Baseline", "M5 DL (Corrected)", "M5 DL + QDM"]
    sig_keys = ["SIG_EQM", "SIG_DL", "SIG_M5_DL_QDM"]
    weeks = [1, 2, 3, 4]
    
    n_cols_per_panel = len(weeks) * len(THRESHOLDS) # 12 columns per subplot
    
    print(f"Generating Mega-Heatmap: {save_name}...")

    # first pass: find global max abs difference so the colormap is fair everywhere
    vmax_dict = {m: 0.0 for m in METRICS_TO_PLOT}
    for m_idx, metric in enumerate(METRICS_TO_PLOT):
        lower_is_better = (metric == "FAR")
        for mod in models:
            for t in THRESHOLDS:
                for w in weeks:
                    if w not in weekly_scores: continue
                    for region in REGIONS:
                        if region not in weekly_scores[w][t]: continue
                        
                        data = weekly_scores[w][t][region]
                        v_raw = data["RAW"][METRIC_INDICES[metric]]
                        v_mod = data[mod][METRIC_INDICES[metric]]
                        diff = (v_raw - v_mod) if lower_is_better else (v_mod - v_raw)
                        vmax_dict[metric] = max(vmax_dict[metric], abs(diff))

    # set up the massive figure grid. made it wider to fit the 12 columns gracefully.
    fig, axes = plt.subplots(len(models), len(METRICS_TO_PLOT), figsize=(28, 16), sharex=True, sharey=True)

    for row_idx, mod in enumerate(models):
        for col_idx, metric in enumerate(METRICS_TO_PLOT):
            ax = axes[row_idx, col_idx]
            grid_data = np.zeros((len(REGIONS), n_cols_per_panel))
            sig_grid = np.zeros((len(REGIONS), n_cols_per_panel), dtype=bool)
            
            lower_is_better = (metric == "FAR")

            for t_idx, t in enumerate(THRESHOLDS):
                for w_idx, w in enumerate(weeks):
                    # calculating the exact column position inside the 12-slot panel
                    col_pos = (t_idx * len(weeks)) + w_idx
                    
                    if w not in weekly_scores: continue
                    for r_idx, region in enumerate(REGIONS):
                        if region not in weekly_scores[w][t]: continue
                        
                        data = weekly_scores[w][t][region]
                        v_raw = data["RAW"][METRIC_INDICES[metric]]
                        v_mod = data[mod][METRIC_INDICES[metric]]
                        sig_thresh = data[sig_keys[row_idx]][METRIC_INDICES[metric]]
                        
                        # standardize so positive is ALWAYS an improvement
                        diff = (v_raw - v_mod) if lower_is_better else (v_mod - v_raw)
                        grid_data[r_idx, col_pos] = diff
                        
                        true_diff = v_mod - v_raw
                        sig_grid[r_idx, col_pos] = abs(true_diff) >= sig_thresh

            # diverging colormap (pink = degrade, green = improve)
            cmap = plt.cm.PiYG 
            bound = vmax_dict[metric] + 0.01
            norm = mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
            
            im = ax.imshow(grid_data, cmap=cmap, norm=norm, aspect='auto')
            
            # add significance stars
            for i in range(len(REGIONS)):
                for j in range(n_cols_per_panel):
                    if sig_grid[i, j]:
                        ax.text(j, i, '★', ha='center', va='center', color='black', fontsize=16, fontweight='bold')

            # draw thick black lines to separate the p33, p50, and p66 threshold zones
            ax.axvline(x=3.5, color='black', linewidth=2.0)
            ax.axvline(x=7.5, color='black', linewidth=2.0)

            # formatting the grid boundaries
            ax.set_xticks(np.arange(n_cols_per_panel))
            if row_idx == len(models) - 1:
                # print W1 W2 W3 W4 repetitively across the bottom
                ax.set_xticklabels([f"W{w}" for _ in THRESHOLDS for w in weeks], fontproperties=bold_font, fontsize=11)
                
                # drop the threshold labels clearly underneath their respective blocks
                ax.text(2, -0.15, "p33", transform=ax.get_xaxis_transform(), ha='center', va='top', fontproperties=bold_font, fontsize=14, color='black')
                ax.text(6, -0.15, "p50", transform=ax.get_xaxis_transform(), ha='center', va='top', fontproperties=bold_font, fontsize=14, color='black')
                ax.text(10, -0.15, "p66", transform=ax.get_xaxis_transform(), ha='center', va='top', fontproperties=bold_font, fontsize=14, color='black')
            
            if col_idx == 0:
                ax.set_yticks(np.arange(len(REGIONS)))
                ax.set_yticklabels(REGIONS, fontproperties=region_label_font)
                ax.set_ylabel(model_titles[row_idx], fontproperties=title_font, labelpad=20)
            
            if row_idx == 0:
                ax.set_title(metric, fontproperties=title_font, pad=15)
                
            # colorbars for each column, placed at the very bottom
            if row_idx == len(models) - 1:
                cax = fig.add_axes([ax.get_position().x0, 0.02, ax.get_position().width, 0.02])
                cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label(f"$\Delta$ {metric} (Green = Better)", fontproperties=bold_font)

    # plt.suptitle(f"Unified Performance Portrait: $\Delta$ Skill vs Raw NCUM\n(★ = Significant Change at {CONFIDENCE_PERCENT}% Confidence)", 
                 # fontproperties=title_font, fontsize=28, y=0.98)
    
    # adjust layout to make room for threshold text and bottom colorbars
    plt.subplots_adjust(top=0.90, bottom=0.15, hspace=0.1, wspace=0.1)
    plt.savefig(os.path.join(PLOTS_DIR, f"{save_name}.png"), dpi=600, bbox_inches='tight')
    plt.close()


# ================= 3. DRIVER =================
def run():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    sample_file = os.path.join(DATA_DIR, "L3YO_Results_Week1.nc")
    if not os.path.exists(sample_file):
        print(f"Fatal: Cannot find data directory at {DATA_DIR}")
        return
        
    with netCDF4.Dataset(sample_file, 'r') as ds:
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]
        
    STATIC_MASKS = build_region_masks(lats, lons)
    weekly_scores = {}
    
    n_workers = min(96, mp.cpu_count()) 
    print(f"\n🚀 Booting Multiprocessing Pool with {n_workers} active cores...")
    
    with mp.Pool(n_workers) as pool:
        
        print("\nMapping Stage 1 (NetCDF Extraction & Baseline Metrics)...")
        ext_tasks = [(w, STATIC_MASKS, lats, lons) for w in range(1, 5)]
        ext_results = pool.map(worker_extract_base_data, ext_tasks)
        
        boot_tasks = []
        for res in ext_results:
            if res is None: continue
            week, season_ids, base_scores, ts_data = res
            weekly_scores[week] = base_scores
            
            for tvar in THRESHOLDS:
                for rname in REGIONS:
                    if rname not in ts_data[tvar]: continue
                    dat = ts_data[tvar][rname]
                    
                    for mod_key, sig_key in zip(['eqm', 'dl', 'dl_qdm'], ['SIG_EQM', 'SIG_DL', 'SIG_M5_DL_QDM']):
                        seed = hash(f"{week}_{tvar}_{rname}_{mod_key}") % (2**32)
                        # queue up the 1D bootstraps for every model
                        boot_tasks.append((week, tvar, rname, mod_key, sig_key, dat['obs'], dat['raw'], dat[mod_key], dat['thr'], season_ids, seed))
        
        print(f"\nSpawning {len(boot_tasks)} micro-tasks to saturate CPUs for Significance Bootstrapping...")
        boot_results = pool.map(worker_bootstrap_1d, boot_tasks)
        
        # repopulate significance bounds
        for res in boot_results:
            week, tvar, rname, sig_key, sig_result = res
            weekly_scores[week][tvar][rname][sig_key] = sig_result
            
    print("\nData loaded and bootstrapped successfully. Beginning plotting...")
    
    if weekly_scores:
        # executing the unified mega-plot!
        plot_mega_heatmap(weekly_scores, save_name="Unified_threshold_based_skill_scores_all_regions")

if __name__ == "__main__":
    run()