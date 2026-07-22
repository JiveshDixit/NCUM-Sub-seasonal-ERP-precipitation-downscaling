"""
Subset-Sensitivity Experiment (Multi-Core Optimized)
----------------------------------------------------
Compares the PRODUCTION arm (using the selected predictor subsets) against a 
CONTROL arm (where marginal predictors are swapped for unselected ones, keeping 
the channel count identical).

The Critical Question:
Does swapping the predictor subset move the skill by MORE than simply re-training 
the identical model with a different random seed (the noise floor)? If not, the 
feature selection is merely a parsimony device, not the source of the predictive skill.

Performance Optimizations:
1. Additive metrics (MAE, NMAE, FSS) are precomputed into temporal partial sums.
2. Spearman rank correlation is fully Pandas-vectorized and restricted to land-only pixels.
3. Bootstrapping is distributed across available CPUs using Joblib.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration ---
RESULTS_DIR = "./M5_Final_Results_oro_newer"
OUT_DIR     = "./SUBSET_SENSITIVITY"
NF_JSON     = "./SEED_ENSEMBLE/noise_floor.json"

WEEKS       = [1, 2, 3, 4]
FSS_PCTL    = 66          
FSS_WINDOW  = 5           
N_BOOT      = 1000        
RNG_SEED    = 0
N_JOBS      = int(os.environ.get("COMPARE_N_JOBS", "-1"))

V_OBS   = "imd_precip"
V_FINAL = "m5_dl_qdm_precip"

os.makedirs(OUT_DIR, exist_ok=True)

# --- Optimization Primitives ---
def precompute(fld, obs, valid_mask, thr):
    """
    Reduces 3D additive metrics to 1D arrays of temporal partial sums.
    Allows bootstrap draws to collapse into ultra-fast index selections.
    """
    f = np.where(valid_mask[None, :, :], fld, np.nan)
    o = np.where(valid_mask[None, :, :], obs, np.nan)

    ae = np.abs(f - o)
    ae_row = np.nansum(ae, axis=(1, 2))
    n_row  = np.sum(np.isfinite(ae), axis=(1, 2)).astype(np.float64)
    o_row  = np.nansum(o, axis=(1, 2))

    # Calculate boolean masks for FSS thresholding
    pf = uniform_filter((np.nan_to_num(f, nan=-1e30) >= thr[None, :, :]).astype(np.float64), 
                        size=(1, FSS_WINDOW, FSS_WINDOW), mode='constant', cval=0.0)
    po = uniform_filter((np.nan_to_num(o, nan=-1e30) >= thr[None, :, :]).astype(np.float64), 
                        size=(1, FSS_WINDOW, FSS_WINDOW), mode='constant', cval=0.0)
    
    m3 = np.broadcast_to(valid_mask[None, :, :], pf.shape)
    d2 = np.where(m3, (pf - po) ** 2, 0.0)
    s2 = np.where(m3, pf ** 2 + po ** 2, 0.0)
    
    return dict(ae=ae_row, n=n_row, osum=o_row, fss_num=d2.sum(axis=(1, 2)), fss_den=s2.sum(axis=(1, 2)))

def metrics_from_rows(P, idx):
    """Rapidly evaluate the additive metrics on a resampled set of timestep indices."""
    ae, n = P['ae'][idx].sum(), P['n'][idx].sum()
    osum = P['osum'][idx].sum()
    num, den = P['fss_num'][idx].sum(), P['fss_den'][idx].sum()
    return {
        'MAE':      ae / n if n > 0 else np.nan,
        'NMAE':     ae / osum if osum > 0 else np.nan,
        'FSS(p66)': 1.0 - num / den if den > 0 else np.nan,
    }

def _spearman_chunk(block, sel_by_season, obs_land, fld_ctrl_land, fld_prod_land):
    """
    Pandas-vectorized Spearman correlation executed on land-only pixels.
    Returns array of (control_rho, prod_rho) for the bootstrap draws in this block.
    """
    out = np.empty((len(block), 2))
    
    for i, draw in enumerate(block):
        sel = np.concatenate([sel_by_season[s] for s in draw])
        
        # Rank observed data
        o_rank = pd.DataFrame(obs_land[sel]).rank(axis=0).values
        o_dev = o_rank - np.nanmean(o_rank, axis=0)
        var_o = np.nansum(o_dev ** 2, axis=0)
        
        # Correlate Control Arm
        c_rank = pd.DataFrame(fld_ctrl_land[sel]).rank(axis=0).values
        c_dev = c_rank - np.nanmean(c_rank, axis=0)
        var_c = np.nansum(c_dev ** 2, axis=0)
        cov_c = np.nansum(c_dev * o_dev, axis=0)
        
        # Correlate Production Arm
        p_rank = pd.DataFrame(fld_prod_land[sel]).rank(axis=0).values
        p_dev = p_rank - np.nanmean(p_rank, axis=0)
        var_p = np.nansum(p_dev ** 2, axis=0)
        cov_p = np.nansum(p_dev * o_dev, axis=0)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            rho_c = cov_c / np.sqrt(var_c * var_o)
            rho_p = cov_p / np.sqrt(var_p * var_o)
            
        out[i, 0] = np.nanmean(rho_c)
        out[i, 1] = np.nanmean(rho_p)
        
    return out

def spearman_mean(f_land, o_land):
    """Single-pass vectorized Spearman mean for the point estimate."""
    o_rank = pd.DataFrame(o_land).rank(axis=0).values
    f_rank = pd.DataFrame(f_land).rank(axis=0).values
    
    o_dev = o_rank - np.nanmean(o_rank, axis=0)
    f_dev = f_rank - np.nanmean(f_rank, axis=0)
    
    cov = np.nansum(f_dev * o_dev, axis=0)
    var_f = np.nansum(f_dev ** 2, axis=0)
    var_o = np.nansum(o_dev ** 2, axis=0)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.nanmean(cov / np.sqrt(var_f * var_o)))

# --- I/O Helpers ---
def load_noise_floor():
    if not os.path.exists(NF_JSON):
        raise FileNotFoundError(f"{NF_JSON} missing. Run seed_ensemble_stats.py first.")
    with open(NF_JSON) as fh: nf = json.load(fh)
    print(f"[FLOOR] Measured on n={nf['n_seeds']} seeds")
    for k, v in sorted(nf['floor'].items()): print(f"[FLOOR]   {k:<12s} {v:.4f}")
    return nf['floor'], nf['n_seeds']

def load_arm(week, control):
    suffix = "_CONTROL" if control else ""
    path = os.path.join(RESULTS_DIR, f"GoldStandard_L3YO_Results_Week{week}{suffix}.nc")
    if not os.path.exists(path): return None, None, None
    with xr.open_dataset(path) as ds:
        obs, fin = ds[V_OBS].values.astype(np.float64), ds[V_FINAL].values.astype(np.float64)
        years = ds['t'].dt.year.values
    return obs, fin, years

# --- Main Logic ---
def main():
    t0 = time.time()
    floor, n_seeds = load_noise_floor()
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for week in WEEKS:
        obs_p, prod, years = load_arm(week, control=False)
        obs_c, ctrl, _     = load_arm(week, control=True)
        
        if prod is None: print(f"\n[SKIP] Week {week}: Production file missing."); continue
        if ctrl is None: print(f"\n[SKIP] Week {week}: Control file missing. Run with --control first."); continue
        if not np.allclose(np.nan_to_num(obs_p), np.nan_to_num(obs_c)):
            raise RuntimeError(f"Week {week}: Truth fields differ. Contrast is invalid.")

        tw = time.time()
        obs = obs_p

        valid = (np.isfinite(obs).all(axis=0) & np.isfinite(prod).all(axis=0) & np.isfinite(ctrl).all(axis=0))
        nvalid = int(valid.sum())

        thr = np.full(obs.shape[1:], np.inf)
        thr[valid] = np.percentile(obs[:, valid], FSS_PCTL, axis=0)

        P_prod = precompute(prod, obs, valid, thr)
        P_ctrl = precompute(ctrl, obs, valid, thr)

        o_land, p_land, c_land = obs[:, valid], prod[:, valid], ctrl[:, valid]
        uy = np.unique(years)
        year_idx = {y: np.where(years == y)[0] for y in uy}
        all_idx = np.arange(len(years))

        # 1. Evaluate Point Estimates
        est_p = metrics_from_rows(P_prod, all_idx)
        est_c = metrics_from_rows(P_ctrl, all_idx)
        est_p['Spearman'] = spearman_mean(p_land, o_land)
        est_c['Spearman'] = spearman_mean(c_land, o_land)

        # 2. Resample Block Indices
        draws = [np.concatenate([year_idx[y] for y in rng.choice(uy, size=len(uy), replace=True)]) for _ in range(N_BOOT)]

        # 3. Additive Metrics (Fast Serial Summation)
        add_deltas = {m: np.empty(N_BOOT) for m in ('MAE', 'NMAE', 'FSS(p66)')}
        for b, idx in enumerate(draws):
            a, d = metrics_from_rows(P_ctrl, idx), metrics_from_rows(P_prod, idx)
            for m in add_deltas: add_deltas[m][b] = a[m] - d[m]

        # 4. Spearman Metric (Parallel Joblib execution on Land pixels)
        nw = N_JOBS if N_JOBS > 0 else os.cpu_count()
        blocks = [b for b in np.array_split(draws, max(nw, 1)) if len(b)]
        
        sp_results = np.vstack(Parallel(n_jobs=nw, prefer="processes")(
            delayed(_spearman_chunk)(b, [np.where(years == s)[0] for s in uy], o_land, c_land, p_land) for b in blocks
        ))
        
        boot = dict(add_deltas)
        boot['Spearman'] = sp_results[:, 0] - sp_results[:, 1] # Control - Prod

        print(f"\n=== Week {week} ({nvalid} land cells, {len(uy)} seasons, {time.time()-tw:.1f}s) ===")
        for name in ('MAE', 'NMAE', 'Spearman', 'FSS(p66)'):
            delta = est_c[name] - est_p[name]
            lo, hi = np.percentile(boot[name], [5, 95])
            fl = floor.get(name, np.nan)
            ratio = abs(delta) / fl if np.isfinite(fl) and fl > 0 else np.nan
            exceeds = bool(np.isfinite(ratio) and ratio > 1.0)
            
            rows.append(dict(week=week, metric=name, production=round(est_p[name], 4), control=round(est_c[name], 4),
                             delta=round(delta, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), noise_floor=round(fl, 4),
                             abs_delta_over_floor=round(ratio, 3), exceeds_noise_floor=exceeds))
                             
            print(f"  {name:<10s} Prod={est_p[name]:9.4f}  Ctrl={est_c[name]:9.4f}  Delta={delta:+8.4f}  Floor={fl:.4f}  Ratio={ratio:5.2f} -> {'EXCEEDS FLOOR' if exceeds else 'within floor'}")

    if not rows: print("\nNothing to compare."); return

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "subset_contrasts.csv")
    df.to_csv(out, index=False)

    n_tot, n_exc = len(df), int(df['exceeds_noise_floor'].sum())
    expected = 0.10 * n_tot 

    print(f"\n{'='*80}\n  SUBSET-SENSITIVITY RESULT\n{'='*80}")
    print(f"  Combinations tested                    : {n_tot}")
    print(f"  Exceeding the noise floor              : {n_exc}")
    print(f"  Expected by chance under the null      : {expected:.1f}")
    print(f"  Largest |delta| as a multiple of floor : {df['abs_delta_over_floor'].max():.2f}x\n")

if __name__ == "__main__":
    main()