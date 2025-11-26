import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    
    """Load dataset and basic cleaning"""
    df = pd.read_csv(path)
    
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="ignore")
            df.sort_values(by="Date", inplace=True)
        except Exception:
            pass

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna(method="ffill", inplace=True)  
    df.reset_index(drop=True, inplace=True)
    return df

