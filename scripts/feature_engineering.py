import pandas as pd

def create_feature_dataset(input_path, output_path):
    """
    Loads the cleaned dataset, normalizes all monetary values to Toman,
    reorders columns, and saves the final feature-ready dataset.
    
    Args:
        input_path (str): Path to the cleaned CSV file (e.g., 'cleaned_gold_dataset.csv').
        output_path (str): Path to save the final feature-engineered CSV file.
    """
    print(f"--- Feature Engineering Started: Loading {input_path} ---")
    
    # 1. Load the cleaned and imputed dataset
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("Initial columns and data types:\n", df.info())
    print("\nOriginal Data Sample:\n", df.head())

    # 2. Normalize currencies to Toman
    # Assumption: USD_IRR from TGJU is the price in Rial, like Gold_IRR.
    print("\nStep 2: Normalizing all prices to Toman...")
    
    # Convert Rial to Toman by dividing by 10
    df['Gold_Toman'] = df['Gold_IRR'] / 10
    df['USD_Toman'] = df['USD_IRR'] / 10
    
    # Convert USD-based assets to Toman by multiplying with the daily USD_Toman rate
    df['Ounce_Toman'] = df['Ounce_USD'] * df['USD_Toman']
    df['Oil_Toman'] = df['Oil_USD'] * df['USD_Toman']
    
    print("   ✅ Currency normalization complete.")

    # 3. Select and reorder columns for the final dataset
    # The 'Date' column is moved to the front, followed by the new Toman-based features.
    final_columns = ['Date', 'Gold_Toman', 'USD_Toman', 'Ounce_Toman', 'Oil_Toman']
    df_features = df[final_columns]
    
    print("\nStep 3: Columns reordered and selected.")

    # 4. Save the final feature-engineered dataset
    df_features.to_csv(output_path, index=False, float_format='%.2f')
    
    print(f"\n✅ FEATURE ENGINEERING COMPLETE! File saved to: {output_path}")
    print("\nFinal Feature-Engineered Data Sample:\n", df_features.head())

if __name__ == "__main__":
    # Define input from the previous step and output for the modeling step
    cleaned_file = "cleaned_gold_dataset.csv"
    features_file = "final_features_toman.csv"
    
    create_feature_dataset(cleaned_file, features_file)
