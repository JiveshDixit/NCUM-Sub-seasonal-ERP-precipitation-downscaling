# Two-stage deep learning for subseasonal Indian monsoon rainfall

Source code for the manuscript *Enhancing Subseasonal Predictability of Indian
Monsoon Rainfall Using Deep Learning and Variance-Restoring Post-Processing*
(Dixit et al., in review at GRL).

## Layout

```
feature_selection/    XGBoost predictor selection -> Table S1
pipeline/             training, inference, seed replicates, control arm
scoring/              reads .nc predictions, writes tables
ablation/             architectural ablation -> Table S2
figures/              reads .nc predictions, writes .png
```

## Data

Two external datasets are required. Neither is redistributed here.

- **IMD 0.25° gridded rainfall**, JJAS 1993 to 2015, weekly aggregates named
  `IMD_week*_sum_25km_on_model_t.nc`. From the India Meteorological Department
  Pune (`https://imdpune.gov.in/`).
- **NCUM extended-range hindcast** is not publicly hosted; access can be requested
  from the Head, NCMRWF (`https://www.nwp.ncmrwf.gov.in/`).

Set the data paths at the top of the pipeline and feature-selection scripts
before running anything.

## Environment

Python 3.11 

Training uses TensorFlow with a CUDA-capable GPU. All scoring and figure
scripts run on CPU.

## Reproducing the paper

Numbered by pipeline order. Each block reproduces one item.

```bash
# 1. Predictor selection -> Table S1, Figure S1
python feature_selection/Xgb_feature_importance_new.py
python feature_selection/final_selected_features.py

# 2. Production training, one command per lead week
for w in 1 2 3 4; do
    python pipeline/Precipitation_correction_sub_seasonal_imd_qdm_no_blending.py --week $w
done

# 3. Table 1
python scoring/headline_skill.py

# 4. Seed ensemble -> Table S3
#    Ten training runs with seeds 43-52, one per node with four GPUs each.
#    On PBS this is submit_all_seeds.sh; on any other scheduler, write the
#    equivalent loop.
for s in 43 44 45 46 47 48 49 50 51 52; do
    for w in 1 2 3 4; do
        python pipeline/Precipitation_correction_sub_seasonal_imd_qdm_no_blending.py \
            --week $w --seed $s
    done
done
python scoring/seed_ensemble_stats.py

# 5. Subset-sensitivity control arm -> Table S4
for w in 1 2 3 4; do
    python pipeline/Precipitation_correction_sub_seasonal_imd_qdm_no_blending.py \
        --week $w --control
done
export COMPARE_N_JOBS=64
python scoring/compare_subset_arms.py

# 6. Architectural ablation -> Table S2
python ablation/Abalation_study.py

# 7. Threshold-leak check -> the 0.37% number in Text S1
python scoring/Obs_threshold_new.py

# 8. Figures 2, 3, 4, 5
python figures/Validation_plots_all_weeks_final.py
python figures/spatial_spectra_variance_check.py
python figures/Unified_threshold_based_skill_scores.py
```

## Reproducibility

GPU training is not bit-reproducible: cuDNN accumulates gradients in a
non-deterministic order. A single re-training reproduces the published skill
to within the noise floor of Table S3, which is 1.11 mm week⁻¹ (MAE), 0.023
(Spearman), 0.015 (FSS at p66) and 0.025 (NMAE), worst case across lead weeks.

## Citation

<!-- If you use this code, please cite:

Dixit, J. et al. *Enhancing Subseasonal Predictability of Indian Monsoon
Rainfall Using Deep Learning and Variance-Restoring Post-Processing.*
Geophysical Research Letters, in review. -->

<!-- Zenodo archive: `<INSERT ZENODO DOI>`. -->

## Contact

`jiveshdixit@gmail.com`
