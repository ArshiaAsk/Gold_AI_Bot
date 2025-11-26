import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_financial_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from Gold USD Oil dataset"""
    
    scaler = StandardScaler()
    scaled_df = df.copy()
    
    for col in ['Gold', 'USD', 'Oil']:
        if col in scaled_df.columns:
            scaled_df[col] = np.log(scaled_df[col])
            
    inputs = [c for c in ['USD', 'Oil']]
    scaled_inputs = scaler.fit_transform(scaled_df[inputs])
    scaled_df[inputs] = scaled_inputs      
    
    df.dropna(inplace=True)

    return scaled_df, scaler
    
    


