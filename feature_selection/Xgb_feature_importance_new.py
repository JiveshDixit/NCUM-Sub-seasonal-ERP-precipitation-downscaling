"""
xgboost feature importance script for the monsoon downscaling.
this calculates permutation importance to see which predictors 
matter the most for each lead week. 

i added a stability loop (n_seeds) so the results are robust and not just a lucky draw, 
and an audit to track which features get dropped by the redundancy filter.
"""

import os
import gc
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# basic path configuration
PREDICTOR_DIR  = os.path.expanduser("/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/erfgc2/final_weekly_timeseries")
OROGRAPHY_FILE = "/home/ncmrwf/bcwc/jivesh/HINDCAST_DATA_ERP/new/geopotential.nc"
PREDICTOR_PATTERN = "Week_{week}_AllYears.nc"
TARGET_DIR     = os.path.expanduser("~/hindcast_erp/Obs_precip")
TARGET_PATTERN = "IMD_week{week}_sum_25km_on_model_t.nc"
LEAD_WEEKS  = [1, 2, 3, 4]
RESULTS_DIR = "./FEATURE_IMPORTANCE_RESULTS_ROBUST_REDUNDANCY_False"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# robustness controls for the training loop
N_SEEDS              = 10        # repeated train/test splits for stability
TEST_SIZE            = 0.25
PERM_REPEATS         = 10        # repeats for permutation importance
PERM_TEST_CAP        = 20000     # cap test rows so permutation doesn't take forever
MAX_WORKING          = 150000    # cap precipitating sample pool before the loop
PRECIP_THRESHOLD     = 1.0       # mm/week; guards against zero-inflation

# critical test switch:
# true -> reproduces the current per-week feature sets (filter drops collinear members).
# false -> keeps all features so weeks share a common set. if the transition still 
#          holds here, it's actual physics and not just bookkeeping.
APPLY_REDUNDANCY_FILTER = False

# calculating physical variables
def calculate_ivt(ds):
    g = 9.80665
    if not all(var in ds for var in ['q', 'u', 'v']):
        return None
    try:
        p_coord = next((c for c in ['p', 'level', 'isobaricInhPa', 'pressure'] if c in ds.coords), None)
        if p_coord is None:
            return None
        p_factor = 100.0 if ds[p_coord].max() > 2000 else 1.0
        q, u, v = ds['q'], ds['u'], ds['v']
        qu_int = (q * u).integrate(coord=p_coord) * (p_factor / g)
        qv_int = (q * v).integrate(coord=p_coord) * (p_factor / g)
        return np.sqrt(qu_int**2 + qv_int**2)
    except Exception:
        return None


def calculate_dynamic_physics(ds):
    lat_key = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_key = 'longitude' if 'longitude' in ds.coords else 'lon'
    dy = 111000.0
    lat_rad = np.deg2rad(ds.coords[lat_key])
    dx = 111000.0 * np.cos(lat_rad)
    try:
        target_p = 85000 if ds.p.max() > 2000 else 850
        q850 = ds['q'].sel(p=target_p, method='nearest')
        u850 = ds['u'].sel(p=target_p, method='nearest')
        v850 = ds['v'].sel(p=target_p, method='nearest')
        div_flux = ((q850 * u850).differentiate(lon_key) / dx) + \
                   ((q850 * v850).differentiate(lat_key) / dy)
        return -div_flux
    except Exception:
        return None


def calculate_derived_physics(ds):
    derived = []
    if 't' in ds.coords:
        month = ds.t.dt.month
        cos_m = np.cos(2 * np.pi * month / 12.0)
        sin_m = np.sin(2 * np.pi * month / 12.0)
        master_var = ds['tot_precip']
        if 'channel' in master_var.dims:
            master_var = master_var.isel(channel=0)
        zeros_grid = xr.zeros_like(master_var)
        derived.append((zeros_grid + cos_m).assign_coords(channel='month_cos'))
        derived.append((zeros_grid + sin_m).assign_coords(channel='month_sin'))
    return derived


# data loading and preprocessing
def load_and_preprocess(p_fp, t_fp, orog_da=None, apply_filter=True):
    print(f"Loading predictors: {os.path.basename(p_fp)}")

    def standardize_coords(ds_or_da):
        rename_map = {}
        if 'lat' in ds_or_da.coords: rename_map['lat'] = 'latitude'
        if 'lon' in ds_or_da.coords: rename_map['lon'] = 'longitude'
        return ds_or_da.rename(rename_map) if rename_map else ds_or_da

    def open_nc(path):
        for eng in ['netcdf4', 'h5netcdf', 'scipy']:
            try:
                return xr.open_dataset(path, engine=eng, decode_times=True, lock=False)
            except Exception:
                continue
        raise IOError(f"Could not open {path}")

    p_ds = standardize_coords(open_nc(p_fp))
    t_ds = standardize_coords(open_nc(t_fp))

    Y_obs = t_ds['tp_weekly_sum']
    if Y_obs.max() < 10.0: Y_obs = Y_obs * 1000.0
    Y_obs = Y_obs.fillna(0.0)

    p_dates = pd.to_datetime(p_ds.t.values).normalize()
    y_dates = pd.to_datetime(Y_obs.t.values).normalize()
    df_p = pd.DataFrame({'p_time': p_dates, 'p_idx': np.arange(len(p_dates))}).sort_values('p_time')
    df_y = pd.DataFrame({'y_time': y_dates, 'y_idx': np.arange(len(y_dates))}).sort_values('y_time')
    merged = pd.merge_asof(df_y, df_p, left_on='y_time', right_on='p_time',
                           direction='nearest', tolerance=pd.Timedelta('3d')).dropna()
    p_ds = p_ds.isel(t=merged['p_idx'].values.astype(int))
    Y_obs = Y_obs.isel(t=merged['y_idx'].values.astype(int))
    p_ds = p_ds.assign_coords(t=Y_obs.t)

    X_list = []

    def add_feature(da, name):
        if da is not None:
            da = standardize_coords(da)
            for d in ['surface', 'ht', 'valid_time', 'number', 'expver', 'toa']:
                if d in da.dims: da = da.squeeze(d, drop=True)
            if 't' not in da.dims:
                da, _ = xr.broadcast(da, p_ds['t'])
            X_list.append(da.fillna(0.0).assign_coords(channel=name))

    add_feature(p_ds['tot_precip'], 'raw_model_precip')
    if 'q' in p_ds:
        is_pa = p_ds.p.max() > 2000
        for lev in [925, 850, 500]:
            target = lev * 100.0 if is_pa else lev
            try: add_feature(p_ds['q'].sel(p=target, method='nearest').drop_vars('p'), f'q_{lev}')
            except Exception: pass
    add_feature(calculate_ivt(p_ds), 'ivt')
    add_feature(calculate_dynamic_physics(p_ds), 'mfc_850')
    if 'olr' in p_ds: add_feature(p_ds['olr'], 'olr')
    for d in calculate_derived_physics(p_ds):
        add_feature(d, str(d.channel.values))
    if 'temp' in p_ds:
        t_var = p_ds['temp']
        add_feature(t_var.isel(ht=0) if 'ht' in t_var.dims else t_var, 't_1p5')
    if 'sm' in p_ds:
        sm = p_ds['sm']
        add_feature(sm.mean('level6') if 'level6' in sm.dims else sm, 'sm')
    for lev in [850, 200]:
        is_pa = p_ds.p.max() > 2000
        target = lev * 100.0 if is_pa else lev
        if 'u' in p_ds: add_feature(p_ds['u'].sel(p=target, method='nearest').drop_vars('p'), f'u_{lev}')
        if 'v' in p_ds: add_feature(p_ds['v'].sel(p=target, method='nearest').drop_vars('p'), f'v_{lev}')
    if 'ht_1' in p_ds:
        is_pa = p_ds.p.max() > 2000
        target_z = 500 * 100.0 if is_pa else 500
        try: add_feature(p_ds['ht_1'].sel(p=target_z, method='nearest').drop_vars('p'), 'z_500')
        except Exception: pass
    if orog_da is not None:
        try:
            target_grid = p_ds['tot_precip'].isel(t=0).squeeze()
            orog_interp = orog_da.interp_like(target_grid, method='nearest',
                                              kwargs={'fill_value': 'extrapolate'})
            add_feature(orog_interp, 'orography_ht')
        except Exception as e:
            print(f"[WARNING] Orography interpolation failed: {e}")

    X = xr.concat(X_list, dim='channel').transpose('t', 'channel', 'latitude', 'longitude')
    predrop_list = X.channel.values.tolist()
    dropped = []

    # removing redundant features if the filter is turned on
    if apply_filter:
        try:
            sample = X.isel(t=0).stack(z=('latitude', 'longitude'))
            df = pd.DataFrame(sample.values.T, columns=X.channel.values)
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            potential_drops = [col for col in upper.columns if any(upper[col] > 0.95)]
            protected = ['ivt', 'mfc_850', 'raw_model_precip', 'orography_ht']
            dropped = [c for c in potential_drops if c not in protected]
            if dropped:
                print(f"[INFO] Redundancy filter dropping: {dropped}")
                X = X.drop_sel(channel=dropped)
        except Exception:
            pass
    else:
        print("[INFO] Redundancy filter DISABLED (common-feature-set test).")

    X = X.interp_like(Y_obs, method='linear', kwargs={'fill_value': 'extrapolate'})
    X, Y_obs = xr.align(X, Y_obs, join='inner')
    final_list = X.channel.values.tolist()
    return X.fillna(0.0), Y_obs, final_list, predrop_list, dropped


# calculating feature importances with a stability loop
def compute_importances(X_s, Y_s, years_s, feats, n_seeds=N_SEEDS):
    """
    repeated train/test fits. returns dict of arrays for gini and permutation importance.
    """
    nf = len(feats)
    gini   = np.full((n_seeds, nf), np.nan)
    perm   = np.full((n_seeds, nf), np.nan)

    unique_years = np.unique(years_s)
    n_test_years = max(1, int(len(unique_years) * TEST_SIZE))

    for si in range(n_seeds):
        # 1. randomly select years to hold out for this specific seed
        rng = np.random.RandomState(si)
        test_years = rng.choice(unique_years, size=n_test_years, replace=False)
        
        # 2. create train/test masks based on the selected years
        test_mask = np.isin(years_s, test_years)
        Xtr, Xte = X_s[~test_mask], X_s[test_mask]
        Ytr, Yte = Y_s[~test_mask], Y_s[test_mask]

        model = xgb.XGBRegressor(n_estimators=100, max_depth=6,
                                 n_jobs=-1, random_state=si)
        model.fit(Xtr, Ytr)

        # gini / impurity
        gini[si] = model.feature_importances_

        # permutation importance on held-out test
        if len(Xte) > PERM_TEST_CAP:
            idx = np.random.RandomState(si).choice(len(Xte), PERM_TEST_CAP, replace=False)
            Xpe, Ype = Xte[idx], Yte[idx]
        else:
            Xpe, Ype = Xte, Yte
        pr = permutation_importance(model, Xpe, Ype, n_repeats=PERM_REPEATS,
                                    random_state=si, n_jobs=-1)
        perm[si] = pr.importances_mean

        print(f"    seed {si+1}/{n_seeds} done "
              f"(train={len(Xtr)}, test={len(Xte)})")
        del model
        gc.collect()

    return dict(gini=gini, perm=perm)


def summarize(feats, imp):
    """builds a dataframe summarizing the mean and standard deviation for each metric."""
    def ms(a):
        return np.nanmean(a, axis=0), np.nanstd(a, axis=0)
    g_m, g_s = ms(imp['gini'])
    p_m, p_s = ms(imp['perm'])
    
    df = pd.DataFrame({
        'feature': feats,
        'gini_mean': g_m, 'gini_std': g_s,
        'perm_mean': p_m, 'perm_std': p_s,
    })
    return df.sort_values('perm_mean', ascending=False).reset_index(drop=True)


# main execution block
def main():
    # pre-load the orography data
    orog_data = None
    if os.path.exists(OROGRAPHY_FILE):
        try:
            with xr.open_dataset(OROGRAPHY_FILE) as ds_o:
                raw = ds_o['z'] / 9.80665 if 'z' in ds_o else \
                      (ds_o['ht'] / 9.80665 if 'ht' in ds_o else ds_o['hgt'])
                ren = {}
                if 'lat' in raw.coords: ren['lat'] = 'latitude'
                if 'lon' in raw.coords: ren['lon'] = 'longitude'
                if ren: raw = raw.rename(ren)
                if 'valid_time' in raw.dims: raw = raw.isel(valid_time=0)
                if 't' in raw.dims: raw = raw.mean('t')
                orog_data = raw.load()
            print("[INIT] Orography loaded.")
        except Exception as e:
            print(f"[ERROR] Orography load failed: {e}")

    long_rows = []
    presence_rows = []

    for week in LEAD_WEEKS:
        print(f"\n================ WEEK {week} ================")
        p_fp = os.path.join(PREDICTOR_DIR, PREDICTOR_PATTERN.format(week=week))
        t_fp = os.path.join(TARGET_DIR, TARGET_PATTERN.format(week=week))
        if not (os.path.exists(p_fp) and os.path.exists(t_fp)):
            print(f"[WARNING] Missing data for Week {week}. Skipping.")
            continue

        X_xr, Y_xr, feats, predrop, dropped = load_and_preprocess(
            p_fp, t_fp, orog_da=orog_data, apply_filter=APPLY_REDUNDANCY_FILTER)

        # auditing the predictor sets
        print(f"  Candidate features (pre-filter): {predrop}")
        print(f"  Dropped by redundancy filter   : {dropped}")
        print(f"  Final feature set ({len(feats)}): {feats}")
        for f in predrop:
            presence_rows.append(dict(week=week, feature=f,
                                      present=int(f in feats),
                                      dropped=int(f in dropped)))

        # setting up the pixel matrix and zero-inflation mask
        X_np = X_xr.values.transpose(0, 2, 3, 1).reshape(-1, len(feats))
        Y_np = Y_xr.values.reshape(-1)
        
        # getting the years and flattening them to match the matrices
        times = pd.to_datetime(Y_xr.t.values)
        years_3d = np.broadcast_to(times.year.values[:, None, None], Y_xr.shape)
        years_np = years_3d.reshape(-1)

        raw_idx = feats.index('raw_model_precip') if 'raw_model_precip' in feats else 0
        mask = (Y_np > PRECIP_THRESHOLD) | (X_np[:, raw_idx] > PRECIP_THRESHOLD)
        
        X_s, Y_s, years_s = X_np[mask], Y_np[mask], years_np[mask]
        
        if len(Y_s) > MAX_WORKING:
            idx = np.random.RandomState(0).choice(len(Y_s), MAX_WORKING, replace=False)
            X_s, Y_s, years_s = X_s[idx], Y_s[idx], years_s[idx] # make sure years stay synced
            
        print(f"  Precipitating samples used: {len(Y_s)}")

        # running the stability loop
        imp = compute_importances(X_s, Y_s, years_s, feats)
        df = summarize(feats, imp)
        df.insert(0, 'week', week)
        df.to_csv(os.path.join(RESULTS_DIR, f"imp_week{week}.csv"), index=False)

        print(f"  Top 5 by permutation importance:")
        print(df[['feature', 'perm_mean', 'perm_std']].head(5).to_string(index=False))

        for _, r in df.iterrows():
            long_rows.append(dict(week=week, feature=r['feature'],
                                  gini=r['gini_mean'], perm=r['perm_mean']))

        del X_xr, Y_xr, X_np, Y_np, X_s, Y_s, imp
        gc.collect()

    # saving the combined output tables
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(RESULTS_DIR, "importance_long_allweeks.csv"), index=False)

    pres = pd.DataFrame(presence_rows)
    pres_mat = pres.pivot_table(index='feature', columns='week', values='present', fill_value=0)
    pres_mat.to_csv(os.path.join(RESULTS_DIR, "predictor_presence_matrix.csv"))
    print("\n=== PREDICTOR PRESENCE MATRIX (1=in model, 0=absent/dropped) ===")
    print(pres_mat.to_string())
    print("\nif a feature is present in some weeks but dropped in others, the")
    print("apparent transition for that feature might be a redundancy filter artifact.")
    print("re-run with APPLY_REDUNDANCY_FILTER=False to confirm with a common set.")

    print(f"\nAll outputs saved to {RESULTS_DIR}")
    print(f"Filter mode: APPLY_REDUNDANCY_FILTER={APPLY_REDUNDANCY_FILTER}")


if __name__ == "__main__":
    main()