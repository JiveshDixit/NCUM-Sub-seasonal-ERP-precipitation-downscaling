import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

# ============================================================
# 0. CONFIGURATION & STYLING
# ============================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
                     'figure.titlesize': 16, 'font.family': 'sans-serif', 'axes.grid': False})

VALIDATION_DIR = "M5_Final_Results_oro_newer"
PLOTS_DIR = "M5_Final_plots_oro_final"

def get_fill_value(var):
    if hasattr(var, '_FillValue'): return var._FillValue
    if hasattr(var, 'missing_value'): return var.missing_value
    if var.dtype.kind == 'f': return np.nan
    return None

def create_masked_array(data, fill_value):
    if fill_value is not None:
        if np.isnan(fill_value): return np.ma.masked_invalid(data)
        else:
            if np.issubdtype(data.dtype, np.floating): return np.ma.masked_where(np.isclose(data, fill_value), data)
            else: return np.ma.masked_equal(data, fill_value)
    return np.ma.masked_array(data, mask=np.zeros_like(data, dtype=bool))

def force_shape_to_time_lat_lon(arr, lat_size, lon_size):
    arr = arr.squeeze()
    if arr.shape[-2] != lat_size or arr.shape[-1] != lon_size:
        if arr.shape[-2] == lon_size and arr.shape[-1] == lat_size:
            arr = arr.transpose(0, 2, 1) 
    return arr.reshape(-1, lat_size, lon_size)

# ============================================================
# 1. SPECTRUM CALCULATION
# ============================================================
def calculate_spatial_power_spectrum(data_ma):
    t, ny, nx = data_ma.shape
    all_spectra = []
    
    for i in range(t):
        frame = np.ma.filled(data_ma[i], 0.0)
        f_transform = np.fft.fft2(frame)
        f_shift = np.fft.fftshift(f_transform)
        power_spec = np.abs(f_shift)**2
        
        y, x = np.indices(power_spec.shape)
        center = (ny // 2, nx // 2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        tbin = np.bincount(r.ravel(), power_spec.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)
        
        max_radius = min(center[0], center[1])
        all_spectra.append(radial_profile[:max_radius])
        
    return np.mean(all_spectra, axis=0)

# ============================================================
# 2. MAIN EXECUTION
# ============================================================
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    spatial_arrays = {'imd': [], 'raw': [], 'eqm': [], 'unblended': [], 'corr': [], 'corr_qdm': [], 'unbcorr_qdm': []}
    lats, lons = None, None

    print("Loading datasets...")
    for w in [1, 2, 3, 4]:
        fname = os.path.join(VALIDATION_DIR, f"L3YO_Results_Week{w}.nc")
        if not os.path.exists(fname): continue
            
        with netCDF4.Dataset(fname, 'r') as ds:
            if lats is None: lats, lons = ds.variables['latitude'][:], ds.variables['longitude'][:]
            
            spatial_arrays['imd'].append(force_shape_to_time_lat_lon(create_masked_array(ds.variables['imd_precip'][:], get_fill_value(ds.variables['imd_precip'])), len(lats), len(lons)))
            spatial_arrays['raw'].append(force_shape_to_time_lat_lon(create_masked_array(ds.variables['raw_model_precip'][:], get_fill_value(ds.variables['raw_model_precip'])), len(lats), len(lons)))
            spatial_arrays['eqm'].append(force_shape_to_time_lat_lon(create_masked_array(ds.variables['eqm_baseline_precip'][:], get_fill_value(ds.variables['eqm_baseline_precip'])), len(lats), len(lons)))
            spatial_arrays['unblended'].append(force_shape_to_time_lat_lon(create_masked_array(ds.variables['dl_only_precip'][:], get_fill_value(ds.variables['dl_only_precip'])), len(lats), len(lons)))
            # spatial_arrays['corr'].append(force_shape_to_time_lat_lon(create_masked_array(ds.variables['optimal_blended_precip'][:], get_fill_value(ds.variables['optimal_blended_precip'])), len(lats), len(lons)))
            # spatial_arrays['corr_qdm'].append(force_shape_to_time_lat_lon(create_masked_array(ds.variables['m5_qdm_precip'][:], get_fill_value(ds.variables['m5_qdm_precip'])), len(lats), len(lons)))
            spatial_arrays['unbcorr_qdm'].append(force_shape_to_time_lat_lon(create_masked_array(ds.variables['m5_dl_qdm_precip'][:], get_fill_value(ds.variables['m5_dl_qdm_precip'])), len(lats), len(lons)))

    if len(spatial_arrays['imd']) == 4:
        filename = os.path.join(PLOTS_DIR, "Spatial_Power_Spectrum_Final.png")
        print(f"Generating Variance Restoration Spectra: {filename}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
        axes = axes.flatten()
        
        for i, w in enumerate([1, 2, 3, 4]):
            ax = axes[i]
            
            spec_imd = calculate_spatial_power_spectrum(spatial_arrays['imd'][i])
            spec_raw = calculate_spatial_power_spectrum(spatial_arrays['raw'][i])
            spec_eqm_base = calculate_spatial_power_spectrum(spatial_arrays['eqm'][i])
            spec_unblended = calculate_spatial_power_spectrum(spatial_arrays['unblended'][i])
            # spec_m5_blend = calculate_spatial_power_spectrum(spatial_arrays['corr'][i])
            # spec_m5_eqm = calculate_spatial_power_spectrum(spatial_arrays['corr_qdm'][i])
            spec_m5_unbqdm = calculate_spatial_power_spectrum(spatial_arrays['unbcorr_qdm'][i])

            
            wavenumbers = np.arange(1, len(spec_imd) + 1)
            
            ax.loglog(wavenumbers, spec_imd, color='#333333', linewidth=3.5, label='IMD (Obs)')
            ax.loglog(wavenumbers, spec_raw, color='#d62728', linewidth=2.5, linestyle='--', label='Raw NCUM')
            ax.loglog(wavenumbers, spec_eqm_base, color='#2ca02c', linewidth=2.5, linestyle='-.', label='EQM Baseline (Raw)')
            ax.loglog(wavenumbers, spec_unblended, color='blueviolet', linewidth=3, linestyle=':', label='M5 (DL Only)')
            ax.loglog(wavenumbers, spec_m5_unbqdm, color='blue', linewidth=3.5, label='M5 + QDM')
            # ax.loglog(wavenumbers, spec_m5_blend, color='#1f77b4', linewidth=2, label='M5 Blended')
            # ax.loglog(wavenumbers, spec_m5_eqm, color='#ff7f0e', linewidth=3, label='M5 Blended + QDM')
            
            ax.set_title(f"Week {w}", fontsize=18, fontweight='bold', pad=10)
            ax.set_xlabel("Wavenumber (Spatial Frequency)", fontsize=14, fontweight='bold')
            ax.set_ylabel("Power Density", fontsize=14, fontweight='bold')
            ax.grid(True, which="both", ls="--", alpha=0.4)
            
            if i == 0:
                ax.legend(fontsize=12, frameon=False, loc='lower left')
                
        plt.suptitle("Spatial Power Spectra: Proof of Variance Restoration", fontsize=24, fontweight='bold', y=0.96)
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        fig.savefig(filename.replace(".png", ".pdf"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
