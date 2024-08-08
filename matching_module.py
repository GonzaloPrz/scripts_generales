import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import LabelEncoder

def estimate_propensity_scores(df, treatment_col, covariate_cols):
    model = LogisticRegression()
    model.fit(df[covariate_cols], df[treatment_col])
    propensity_scores = model.predict_proba(df[covariate_cols])[:, 1]
    return propensity_scores

# Perform nearest neighbor matching
def perform_matching(df, treatment_col, covariate_cols,factor_vars, treatment_value=1,caliper=0.05):
        
    for col in factor_vars:    
        df[col] = LabelEncoder().fit_transform(df[col])
        
    propensity_scores = estimate_propensity_scores(df, treatment_col, covariate_cols)
    
    df['propensity_score'] = propensity_scores
    
    treated = df[df[treatment_col] == treatment_value].reset_index(drop=True)
    control = df[df[treatment_col] != treatment_value].reset_index(drop=True)
    
    n_treated = treated.shape[0]
    n_control = control.shape[0]
    
    if n_treated > n_control:
        matched_data = treated.copy()
        treated = control
        control = matched_data
    else:
        matched_data = control.copy()
    
    #Perform matching without replacement
    nbrs = NearestNeighbors(n_neighbors=control.shape[0],radius=caliper).fit(control[['propensity_score']])
    
    _, indices = nbrs.kneighbors(treated[['propensity_score']])
    
    matched_controls = pd.DataFrame(columns=control.columns)
    
    matched_indices = []
    for i, index in enumerate(indices):
        if i == 0:
            matched_controls.loc[len(matched_controls.index),:] = control.loc[index[0],:]
            matched_indices.append(index[0])
        else:
            for idx in index:
                if idx not in matched_indices:
                    matched_controls.loc[len(matched_controls.index),:] = control.loc[idx,:]
                    matched_indices.append(idx)
                    break
    
    matched_data = pd.concat([treated,matched_controls],axis=0)  
    matched_data = matched_data.reset_index(drop=True) 
    return matched_data