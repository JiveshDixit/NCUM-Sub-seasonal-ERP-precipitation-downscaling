"""
Feature Extraction and Formatting Script
----------------------------------------
Reads XGBoost permutation importance results and extracts the final robust 
predictor subsets for Lead Weeks 1-4. Outputs a padded CSV table suitable 
for manuscript publication or supplementary material.
"""

import os
import pandas as pd

# Directory containing the XGBoost permutation importance CSVs
XGB_RESULTS_DIR = "./FEATURE_IMPORTANCE_RESULTS_ROBUST_REDUNDANCY_False"

def get_robust_features(week, results_dir=XGB_RESULTS_DIR, cumulative_threshold=0.8):
    """
    Extracts the final feature list for a given lead week using three selection rules:
    1. Statistical significance
    2. Cumulative predictive power
    3. Mandatory inclusion of physical downscaling anchors
    """
    csv_path = os.path.join(results_dir, f"imp_week{week}.csv")
    base_features = ['raw_model_precip', 'orography_ht'] 
    
    if not os.path.exists(csv_path):
        print(f"[WARNING] File not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    
    # Rule 1: Filter for statistical significance (Signal > Noise)
    df_sig = df[df['perm_mean'] - df['perm_std'] > 0].copy()
    
    # Rule 2: Keep top features comprising the cumulative importance threshold
    df_sig = df_sig.sort_values('perm_mean', ascending=False)
    df_sig['importance_share'] = df_sig['perm_mean'] / df_sig['perm_mean'].sum()
    df_sig['cumulative_share'] = df_sig['importance_share'].cumsum()
    
    cutoff_idx = df_sig[df_sig['cumulative_share'] >= cumulative_threshold].index
    if len(cutoff_idx) > 0:
        first_breach = df_sig.index.get_loc(cutoff_idx[0])
        selected_features = df_sig.iloc[:first_breach + 1]['feature'].tolist()
    else:
        selected_features = df_sig['feature'].tolist()
        
    # Rule 3: Always include downscaling anchors, even if filtered out above
    for feature in base_features:
        if feature not in selected_features:
            selected_features.append(feature)
            
    return selected_features

def main():
    print("Extracting features and formatting for manuscript...")
    
    features_dict = {}
    max_len = 0
    
    # 1. Gather features for all lead weeks and find the maximum list length
    for week in [1, 2, 3, 4]:
        features = get_robust_features(week)
        features_dict[f'Lead_Week_{week}'] = features
        if len(features) > max_len:
            max_len = len(features)
            
    # 2. Pad shorter lists with empty strings to create a uniform table grid
    for week in [1, 2, 3, 4]:
        key = f'Lead_Week_{week}'
        current_len = len(features_dict[key])
        if current_len < max_len:
            features_dict[key].extend([''] * (max_len - current_len))
            
    # 3. Export the final formatted DataFrame to CSV
    df_final = pd.DataFrame(features_dict)
    out_path = "Final_Selected_Features_Manuscript.csv"
    
    df_final.to_csv(out_path, index=False)
    

if __name__ == "__main__":
    main()