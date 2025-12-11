import pandas as pd
import numpy as np

def clean_and_impute_data(input_path, output_path):
    print(f"--- Cleaning Dataset: {input_path} ---")
    
    # 1. Load dataset from CSV
    df = pd.read_csv(input_path)
    
    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort data chronologically (very important for forward-fill to work correctly)
    df = df.sort_values('Date', ascending=True)
    
    print("\n1. Initial Missing Values:")
    print(df.isnull().sum())
    
    # 2. Handle potential incorrect zero-values
    # If 0.0 exists (due to earlier scrape/API issues), convert it to NaN so it can be imputed later
    cols_to_fix = ['Gold_IRR', 'USD_IRR', 'Ounce_USD', 'Oil_USD']
    for col in cols_to_fix:
        if col in df.columns:
            count_zeros = (df[col] == 0).sum()
            if count_zeros > 0:
                print(f"   Warning: Found {count_zeros} zeros in {col}. Treating as NaN.")
                df[col] = df[col].replace(0, np.nan)

    # 3. Forward Fill Strategy (FFill)
    # Meaning: today’s missing value = last available value before it
    df_clean = df.ffill()
    
    # 4. Backward Fill Strategy (BFill)
    # Used only for very early rows where no "previous" value exists
    df_clean = df_clean.bfill()
    
    print("\n2. Post-Cleaning Missing Values:")
    print(df_clean.isnull().sum())
    
    # 5. Drop rows that still have NaN
    # This happens rarely, only if all columns were empty for a specific date
    original_len = len(df)
    df_clean = df_clean.dropna()
    dropped_len = original_len - len(df_clean)
    if dropped_len > 0:
        print(f"\n   Dropped {dropped_len} rows that were completely unrecoverable.")

    # 6. Final sorting (newest date first)
    df_clean = df_clean.sort_values('Date', ascending=False)
    
    # Save cleaned dataset
    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ CLEAN DATASET SAVED: {output_path}")
    print(df_clean.head())

if __name__ == "__main__":
    clean_and_impute_data("final_gold_dataset.csv", "cleaned_gold_dataset.csv")
