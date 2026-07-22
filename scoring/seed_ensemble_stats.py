"""
Seed Ensemble & Noise Floor Diagnostics (Multi-Core Optimized)
--------------------------------------------------------------
Calculates the expected skill variance caused by training randomness (the Noise Floor).
Outputs two critical files:
1. noise_floor.json: The strict bounds that any paper claim must clear.
2. ensemble_mean_skill.csv: The diagnostic skill of the averaged model ensemble, 
   alongside a spatial smoothing (power spectrum) check.
"""

import os
import glob
import json
import time
import itertools
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration ---
RESULTS_DIR = "./M5_Final_Results_oro_newer"
OUT_DIR     = "./SEED_ENSEMBLE"
WEEKS       = [1, 2, 3, 4]
PRODUCTION_SEED = 42

PRODUCTS = {'M5 Corrected': 'dl_only_precip', 'M5 DL+QDM': 'm5_dl_qdm_precip'}

N_BOOT  = 1000
ALPHA   = 0.10
FSS_Q   = 0.66
FSS_WIN = 5
N_JOBS  = int(os.environ.get("COMPARE_N_JOBS", "-1"))

os.makedirs(OUT_DIR, exist_ok=True)

# --- Helpers ---
def discover_seeds(week):
    """Finds all available seeds for the given lead week."""
    found = {}
    base = os.path.join(RESULTS_DIR, f"GoldStandard_L3YO_Results_Week{week}.nc")
    if os.path.exists(base): found[PRODUCTION_SEED] = base
    
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, f"GoldStandard_L3YO_Results_Week{week}_SEED*.nc"))):
        stem = os.path.basename(p).rsplit("_SEED", 1)[1].replace(".nc", "")
        if stem.isdigit(): found[int(stem)] = p
    return dict(sorted(found.items()))

def per_season_components(fc, obs, season_pos, S, land, thr):
    valid = np.isfinite(obs) & np.isfinite(fc) & land[None, :, :]
    err  = np.where(valid, np.abs(fc - obs), 0.0)
    obs0 = np.where(valid, obs, 0.0)
    fb = np.where(valid, fc  >= thr[None, :, :], 0.0).astype(np.float32)
    ob = np.where(valid, obs >= thr[None, :, :], 0.0).astype(np.float32)
    k = FSS_WIN
    ff = uniform_filter(fb, size=(1, k, k), mode='constant', cval=0.0)
    of = uniform_filter(ob, size=(1, k, k), mode='constant', cval=0.0)
    d2 = np.where(valid, (ff - of) ** 2, 0.0)
    s2 = np.where(valid, ff ** 2 + of ** 2, 0.0)

    def bs(a):
        flat = a.reshape(a.shape[0], -1).sum(axis=1)
        out = np.zeros(S); np.add.at(out, season_pos, flat); return out

    return dict(abs_err=bs(err), obs_sum=bs(obs0), n=bs(valid.astype(np.float64)), fss_num=bs(d2), fss_den=bs(s2))

def scalar_metrics(counts, c):
    ae, nn = counts @ c['abs_err'], counts @ c['n']
    fnum, fden = counts @ c['fss_num'], counts @ c['fss_den']
    return {'MAE': ae / nn, 'NMAE': ae / (counts @ c['obs_sum']), f'FSS(p{int(FSS_Q*100)})': 1.0 - fnum / np.maximum(fden, 1e-9)}

def _spearman_chunk(block, sel_by_season, obs, fields, land):
    """Pandas-vectorized Spearman correlation executed on land-only pixels."""
    names = list(fields)
    out = np.empty((len(block), len(names)))
    
    obs_land = obs[:, land]
    fields_land = {nm: fields[nm][:, land] for nm in names}
    
    for i, draw in enumerate(block):
        sel = np.concatenate([sel_by_season[s] for s in draw])
        o_rank = pd.DataFrame(obs_land[sel]).rank(axis=0).values
        o_dev = o_rank - np.nanmean(o_rank, axis=0)
        var_o = np.nansum(o_dev ** 2, axis=0)
        
        for j, nm in enumerate(names):
            f_rank = pd.DataFrame(fields_land[nm][sel]).rank(axis=0).values
            f_dev = f_rank - np.nanmean(f_rank, axis=0)
            var_f = np.nansum(f_dev ** 2, axis=0)
            
            cov = np.nansum(f_dev * o_dev, axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                corrs = cov / np.sqrt(var_f * var_o)
            
            out[i, j] = np.nanmean(corrs)
    return out

def radial_spectrum(field, land):
    """Mean radially-binned 2D power spectrum to check the spatial smoothing cost."""
    f = np.where(np.isfinite(field), field, 0.0)
    F = np.fft.fftshift(np.fft.fft2(f, axes=(1, 2)), axes=(1, 2))
    P = (np.abs(F) ** 2).mean(axis=0)
    H, W = P.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.hypot(y - cy, x - cx).astype(int)
    nb = min(cy, cx)
    return np.array([P[r == k].mean() for k in range(1, nb)])

# --- Main Driver ---
def main():
    t0 = time.time()
    floor_rows, ens_rows = [], []

    for week in WEEKS:
        seeds = discover_seeds(week)
        if len(seeds) < 2:
            print(f"[SKIP] Week {week}: only {len(seeds)} seed(s) found."); continue

        print(f"\n{'='*84}\n  LEAD WEEK {week}   seeds: {list(seeds)}\n{'='*84}")
        ds = {s: xr.open_dataset(p) for s, p in seeds.items()}
        ref = ds[PRODUCTION_SEED] if PRODUCTION_SEED in ds else next(iter(ds.values()))

        # Integrity Check
        for s, d in ds.items():
            for v in ['imd_precip', 'raw_model_precip', 'eqm_baseline_precip']:
                if not np.allclose(ref[v].values, d[v].values, equal_nan=True, rtol=1e-5, atol=1e-4):
                    raise RuntimeError(f"Integrity Error: Seed {s} changed static field '{v}'.")

        obs  = ref['imd_precip'].values.astype(np.float32)
        land = np.isfinite(obs).any(axis=0)
        years = pd.DatetimeIndex(ref.t.values).year.values
        useasons = np.unique(years); S = len(useasons)
        pos = {s_: i for i, s_ in enumerate(useasons)}
        season_pos = np.array([pos[y] for y in years])
        sel_by_season = [np.where(years == s_)[0] for s_ in useasons]

        with np.errstate(invalid='ignore'):
            thr = np.nanpercentile(obs, 100 * FSS_Q, axis=0)
        thr = np.where(np.isfinite(thr), thr, np.inf).astype(np.float32)

        rng = np.random.default_rng(0)
        draws = rng.integers(0, S, size=(N_BOOT, S))
        counts = np.zeros((N_BOOT, S))
        np.add.at(counts, (np.repeat(np.arange(N_BOOT), S), draws.ravel()), 1.0)

        # Build Fields (Individual Members + Ensemble Mean)
        fields = {}
        for pname, var in PRODUCTS.items():
            for s in seeds: fields[f"{pname}|s{s}"] = ds[s][var].values.astype(np.float32)
            fields[f"{pname}|ENSMEAN"] = np.mean([ds[s][var].values for s in seeds], axis=0).astype(np.float32)

        comp = {k: per_season_components(v, obs, season_pos, S, land, thr) for k, v in fields.items()}
        boots = {k: scalar_metrics(counts, c) for k, c in comp.items()}

        nw = N_JOBS if N_JOBS > 0 else os.cpu_count()
        blocks = [b for b in np.array_split(draws, max(nw, 1)) if len(b)]
        rho = np.vstack(Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(_spearman_chunk)(b, sel_by_season, obs, fields, land) for b in blocks))
        
        names = list(fields)
        for k in fields: boots[k]['Spearman'] = rho[:, names.index(k)]

        metrics = ['MAE', 'NMAE', f'FSS(p{int(FSS_Q*100)})', 'Spearman']

        # Print (1) Noise Floor Spread
        for pname in PRODUCTS:
            print(f"\n  --- {pname}: Across-Seed Spread ---")
            print(f"    {'Metric':<12s} {'Mean':>9s} {'SD':>8s} {'Max Pairwise |Diff|':>20s}")
            for m in metrics:
                vals = np.array([np.nanmedian(boots[f"{pname}|s{s}"][m]) for s in seeds])
                mx = max(abs(a - b) for a, b in itertools.combinations(vals, 2))
                print(f"    {m:<12s} {vals.mean():>9.4f} {vals.std(ddof=1):>8.4f} {mx:>20.4f}")
                floor_rows.append(dict(week=week, product=pname, metric=m, n_seeds=len(seeds), mean=vals.mean(),
                                       sd=vals.std(ddof=1), max_pairwise_diff=mx, values=list(np.round(vals, 5))))

        # Print (2) Ensemble Mean Cost
        for pname in PRODUCTS:
            print(f"\n  --- {pname}: Ensemble Mean vs Members ---")
            for m in metrics:
                em = np.nanmedian(boots[f"{pname}|ENSMEAN"][m])
                mem = np.array([np.nanmedian(boots[f"{pname}|s{s}"][m]) for s in seeds])
                print(f"    {m:<12s} EnsMean={em:>9.4f}   MemberMean={mem.mean():>9.4f}   Delta={em - mem.mean():+.4f}")
                ens_rows.append(dict(week=week, product=pname, metric=m, ensemble_mean=em, member_mean=mem.mean(), delta=em - mem.mean()))

            sp_obs = radial_spectrum(obs, land)
            sp_ens = radial_spectrum(fields[f"{pname}|ENSMEAN"], land)
            sp_mem = np.mean([radial_spectrum(fields[f"{pname}|s{s}"], land) for s in seeds], axis=0)
            hi = slice(len(sp_obs) * 2 // 3, None) 
            r_ens, r_mem = np.nanmean(sp_ens[hi] / sp_obs[hi]), np.nanmean(sp_mem[hi] / sp_obs[hi])
            print(f"    High-k Power / Observed:  Members={r_mem:.3f}   EnsMean={r_ens:.3f}")
            if r_ens < r_mem * 0.95:
                print("    ** WARNING: Averaging seeds physically smooths the field (loses high-k variance). **")

    if floor_rows:
        fd = pd.DataFrame(floor_rows)
        # Establish stable floor: 90% two-sided bound on difference of two independent seeds.
        fd['floor_2seed_90pct'] = 1.645 * np.sqrt(2.0) * fd['sd']

        fd.to_csv(os.path.join(OUT_DIR, "noise_floor.csv"), index=False)
        key = fd[fd['product'] == 'M5 DL+QDM']

        floor = key.groupby('metric')['floor_2seed_90pct'].max().round(6).to_dict()
        n_seeds = int(key['n_seeds'].max())
        
        with open(os.path.join(OUT_DIR, "noise_floor.json"), 'w') as fh:
            json.dump({'n_seeds': n_seeds, 'basis': '1.645*sqrt(2)*SD across seeds', 'floor': floor}, fh, indent=2)
            


    if ens_rows:
        pd.DataFrame(ens_rows).to_csv(os.path.join(OUT_DIR, "ensemble_mean_skill.csv"), index=False)
        


if __name__ == "__main__":
    main()