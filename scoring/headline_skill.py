"""
Headline Skill Evaluator (Multi-Core Optimized)
-----------------------------------------------
Evaluates the four canonical pipeline stages (Raw NCUM, EQM, M5 Corrected, M5 DL+QDM) 
against IMD observations. 

Key Features:
1. Paired seasonal block bootstrap (1000 draws) to measure spatial/temporal significance.
2. Direct comparison against the Training Noise Floor to ensure improvements are 
   driven by architecture/physics, not just random training seeds.
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
OUT_DIR     = "./HEADLINE_SKILL"
WEEKS       = [1, 2, 3, 4]

PRODUCTS = {
    'Raw NCUM':      'raw_model_precip',
    'EQM baseline':  'eqm_baseline_precip',
    'M5 Corrected':  'dl_only_precip',
    'M5 DL+QDM':     'm5_dl_qdm_precip',
}

_FALLBACK = {'MAE': 0.639, 'NMAE': 0.0135, 'FSS(p66)': 0.0178, 'Spearman': 0.0134}
_NF_JSON = "./SEED_ENSEMBLE/noise_floor.json"

# Load the noise floor (Fallback to hardcoded if json missing)
if os.path.exists(_NF_JSON):
    with open(_NF_JSON) as _fh:
        _nf = json.load(_fh)
    NOISE_FLOOR = _nf['floor']
    print(f"[NOISE FLOOR] Measured (n={_nf['n_seeds']} seeds): {NOISE_FLOOR}")
else:
    NOISE_FLOOR = _FALLBACK
    print(f"[NOISE FLOOR] *** FALLBACK (n=2) estimate: {NOISE_FLOOR} ***")

# The specific stage-to-stage comparisons for the manuscript
CONTRASTS = [
    ('EQM baseline', 'Raw NCUM'),      
    ('M5 Corrected', 'Raw NCUM'),      
    ('M5 DL+QDM',    'Raw NCUM'),      
    ('M5 DL+QDM',    'EQM baseline'),  # The paper's core claim
    ('M5 DL+QDM',    'M5 Corrected'),  
]

N_BOOT  = 1000
ALPHA   = 0.10
SEED    = 0
FSS_Q   = 0.66
FSS_WIN = 5
N_JOBS  = int(os.environ.get("COMPARE_N_JOBS", "-1"))

os.makedirs(OUT_DIR, exist_ok=True)
LOWER_BETTER = {'MAE': True, 'NMAE': True, 'FSS(p66)': False, 'Spearman': False}

# --- Core Metric Functions ---
def per_season_components(fc, obs, season_pos, S, land, thr):
    """Reduces 3D fields into seasonal partial sums for lightning-fast bootstrapping."""
    valid = np.isfinite(obs) & np.isfinite(fc) & land[None, :, :]
    err  = np.where(valid, np.abs(fc - obs), 0.0)
    obs0 = np.where(valid, obs, 0.0)
    fc_b  = np.where(valid, fc  >= thr[None, :, :], 0.0).astype(np.float32)
    obs_b = np.where(valid, obs >= thr[None, :, :], 0.0).astype(np.float32)
    
    k = FSS_WIN
    ff = uniform_filter(fc_b,  size=(1, k, k), mode='constant', cval=0.0)
    of = uniform_filter(obs_b, size=(1, k, k), mode='constant', cval=0.0)
    
    d2 = np.where(valid, (ff - of) ** 2, 0.0)
    s2 = np.where(valid, ff ** 2 + of ** 2, 0.0)

    def bs(a):
        flat = a.reshape(a.shape[0], -1).sum(axis=1)
        out = np.zeros(S); np.add.at(out, season_pos, flat); return out

    return dict(abs_err=bs(err), obs_sum=bs(obs0), n=bs(valid.astype(np.float64)), fss_num=bs(d2), fss_den=bs(s2))

def scalar_metrics(counts, c):
    ae, nn = counts @ c['abs_err'], counts @ c['n']
    osum = counts @ c['obs_sum']
    fnum, fden = counts @ c['fss_num'], counts @ c['fss_den']
    return {'MAE': ae / nn, 'NMAE': ae / osum, f'FSS(p{int(FSS_Q*100)})': 1.0 - fnum / np.maximum(fden, 1e-9)}

def _spearman_chunk(block, sel_by_season, obs, fields, land):
    """Pandas-vectorized Spearman correlation executed on land-only pixels."""
    names = list(fields)
    out = np.empty((len(block), len(names)))
    
    # Pre-filter ocean pixels to save memory and compute
    obs_land = obs[:, land]
    fields_land = {nm: fields[nm][:, land] for nm in names}
    
    for i, draw in enumerate(block):
        sel = np.concatenate([sel_by_season[s] for s in draw])
        
        # Rank the observed block
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

def ci(d):
    """Returns median, and 90% confidence intervals."""
    return (np.nanmedian(d), np.nanpercentile(d, 100 * ALPHA / 2), np.nanpercentile(d, 100 * (1 - ALPHA / 2)))

# --- Main Driver ---
def main():
    t0 = time.time()
    rows, point_rows = [], []

    for week in WEEKS:
        f = os.path.join(RESULTS_DIR, f"GoldStandard_L3YO_Results_Week{week}.nc")
        if not os.path.exists(f):
            print(f"[SKIP] Week {week}: {f} not found."); continue

        print(f"\n{'='*84}\n  LEAD WEEK {week}\n{'='*84}")
        ds = xr.open_dataset(f)
        obs  = ds['imd_precip'].values.astype(np.float32)
        land = np.isfinite(obs).any(axis=0)
        years = pd.DatetimeIndex(ds.t.values).year.values
        useasons = np.unique(years); S = len(useasons)
        pos = {s: i for i, s in enumerate(useasons)}
        season_pos = np.array([pos[y] for y in years])
        sel_by_season = [np.where(years == s)[0] for s in useasons]

        with np.errstate(invalid='ignore'):
            thr = np.nanpercentile(obs, 100 * FSS_Q, axis=0)
        thr = np.where(np.isfinite(thr), thr, np.inf).astype(np.float32)

        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, S, size=(N_BOOT, S))
        counts = np.zeros((N_BOOT, S))
        np.add.at(counts, (np.repeat(np.arange(N_BOOT), S), draws.ravel()), 1.0)

        fields = {k: ds[v].values.astype(np.float32) for k, v in PRODUCTS.items()}
        comp = {k: per_season_components(v, obs, season_pos, S, land, thr) for k, v in fields.items()}
        boots = {k: scalar_metrics(counts, c) for k, c in comp.items()}

        # Distribute Spearman ranking to worker pool
        nw = N_JOBS if N_JOBS > 0 else os.cpu_count()
        blocks = [b for b in np.array_split(draws, max(nw, 1)) if len(b)]
        rho = np.vstack(Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(_spearman_chunk)(b, sel_by_season, obs, fields, land) for b in blocks))
        
        names = list(fields)
        for k in fields:
            boots[k]['Spearman'] = rho[:, names.index(k)]

        # Print Absolute Skill
        print(f"\n  {'Product':<15s} {'MAE':>9s} {'NMAE':>8s} {'FSS(p66)':>10s} {'Spearman':>10s}")
        for k in PRODUCTS:
            v = {m: np.nanmedian(b) for m, b in boots[k].items()}
            print(f"  {k:<15s} {v['MAE']:>9.3f} {v['NMAE']:>8.4f} {v['FSS(p66)']:>10.4f} {v['Spearman']:>10.4f}")
            point_rows.append(dict(week=week, product=k, **v))

        # Print Contrasts vs Noise Floor
        print(f"\n  {'Contrast':<32s} {'Metric':<10s} {'Delta':>9s} {'90% CI':>20s} {'Noise':>7s}  Verdict")
        for chal, ref in CONTRASTS:
            for m in ['MAE', 'NMAE', 'FSS(p66)', 'Spearman']:
                d = boots[chal][m] - boots[ref][m]
                md, lo, hi = ci(d)
                nf = NOISE_FLOOR[m]
                sig = (lo > 0) or (hi < 0)
                better = (md < 0) if LOWER_BETTER[m] else (md > 0)
                exceeds = abs(md) > nf

                if not sig: vd = "not significant"
                elif not exceeds: vd = "** BELOW NOISE FLOOR **"
                else: vd = "IMPROVED" if better else "** DEGRADED **"

                label = f"{chal} vs {ref}"
                print(f"  {label:<32s} {m:<10s} {md:+9.4f} [{lo:+7.3f},{hi:+7.3f}] {nf:>7.3f}  {vd}")
                rows.append(dict(week=week, contrast=label, metric=m, delta=md, ci_lo=lo, ci_hi=hi,
                                 noise_floor=nf, block_significant=sig, exceeds_noise=exceeds, verdict=vd))
        ds.close()

    if not rows:
        print("\nNothing computed."); return

    pd.DataFrame(point_rows).to_csv(os.path.join(OUT_DIR, "absolute_skill.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "contrasts_vs_noise.csv"), index=False)

    print(f"\n{'='*84}\n  FINAL VERDICT: M5 DL+QDM vs EQM baseline\n{'='*84}")
    key = df[df['contrast'] == 'M5 DL+QDM vs EQM baseline']
    print(key[['week', 'metric', 'delta', 'ci_lo', 'ci_hi', 'noise_floor', 'verdict']].to_string(index=False))


if __name__ == "__main__":
    main()