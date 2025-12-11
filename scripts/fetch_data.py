from curl_cffi import requests
import pandas as pd
import yfinance as yf
import re
import time
from datetime import datetime

# --- Data Cleaning Utility: Strips HTML tags and sanitizes numerical values ---
def clean_value(x):
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # 1. Strip HTML tags using regex pattern
        x = re.sub(r'<[^>]+>', '', x)
        # 2. Remove currency formatting chars (commas, percent signs)
        x = x.replace(',', '').replace('%', '')
        # 3. Attempt type conversion to float
        try:
            return float(x)
        except ValueError:
            return 0.0
    return 0.0

def get_tgju_history(item_slug):
    # Target Endpoint: TGJU summary-table-data API
    url = f"https://api.tgju.org/v1/market/indicator/summary-table-data/{item_slug}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': f'https://www.tgju.org/profile/{item_slug}/history',
        'Origin': 'https://www.tgju.org',
        'Accept': 'application/json, text/plain, */*',
    }

    print(f"Fetching data for {item_slug} via API...")

    try:
        # Execute HTTP GET with TLS fingerprinting (chrome110) to bypass WAF
        session = requests.Session()
        response = session.get(url, headers=headers, impersonate="chrome110")

        if response.status_code == 200:
            json_data = response.json()
            if 'data' in json_data:
                df = pd.DataFrame(json_data['data'])
                
                # Map API JSON response indices to standard column names
                # 0:Open, 1:Low, 2:High, 3:Close, 4:Change(HTML), 5:Percent(HTML), 6:Date(Gregorian), 7:Date(Jalali)
                cols = ['Open', 'Low', 'High', 'Close', 'Change', 'Percent', 'Date_G', 'Date_J']
                
                # Handle dynamic column length if API response varies
                current_cols = len(df.columns)
                if current_cols >= 8:
                    df.columns = cols + [f'col_{i}' for i in range(8, current_cols)]
                else:
                    df.columns = cols[:current_cols]
                
                # Feature Selection: Retain only Date (Gregorian) and Close Price
                final_df = df[['Date_G', 'Close']].copy()
                
                # Apply data sanitization to 'Close' column
                final_df['Close'] = final_df['Close'].apply(clean_value)
                
                # Normalize date format to YYYY-MM-DD
                final_df['Date'] = pd.to_datetime(final_df['Date_G']).dt.date
                
                # Drop redundant raw date columns
                final_df = final_df.drop(columns=['Date_G'])
                
                # Rename 'Close' column based on asset type for later merging
                col_name = 'Gold_IRR' if 'geram' in item_slug else 'USD_IRR'
                final_df = final_df.rename(columns={'Close': col_name})
                
                print(f"   ✅ Success: {len(final_df)} rows fetched.")
                return final_df
            else:
                print("   ❌ Key 'data' not found in response.")
                return None
        else:
            print(f"   ❌ Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return None

def fetch_global_data():
    print("\nFetching Global Data (Yahoo Finance)...")
    try:
        # Fetch Gold Futures (GC=F)
        gold = yf.Ticker("GC=F")
        df_gold = gold.history(period="5y")[['Close']].rename(columns={'Close': 'Ounce_USD'}).reset_index()
        
        # Fetch Brent Crude Oil (BZ=F)
        oil = yf.Ticker("BZ=F")
        df_oil = oil.history(period="5y")[['Close']].rename(columns={'Close': 'Oil_USD'}).reset_index()
        
        # Normalize datetime objects to date objects for consistency
        df_gold['Date'] = pd.to_datetime(df_gold['Date']).dt.date
        df_oil['Date'] = pd.to_datetime(df_oil['Date']).dt.date
        
        # Outer join global assets on Date
        df_global = pd.merge(df_gold, df_oil, on='Date', how='outer')
        print(f"   ✅ Global data fetched: {len(df_global)} rows")
        return df_global
    except Exception as e:
        print(f"   ❌ Yahoo Error: {e}")
        return None

if __name__ == "__main__":
    # Phase 1: Retrieve Local Market Data (TGJU)
    df_gold = get_tgju_history("geram18")
    df_usd = get_tgju_history("price_dollar_rl")
    
    # Merge Local Assets (Gold + USD)
    main_df = pd.DataFrame()
    if df_gold is not None and df_usd is not None:
        main_df = pd.merge(df_gold, df_usd, on='Date', how='outer')
    elif df_gold is not None:
        main_df = df_gold
    elif df_usd is not None:
        main_df = df_usd
        
    # Phase 2: Retrieve Global Market Data (Yahoo Finance)
    df_global = fetch_global_data()
    
    # Phase 3: Final Dataset Aggregation
    if not main_df.empty and df_global is not None:
        print("\nMerging all datasets...")
        # Left join to preserve local market dates as primary key
        final_dataset = pd.merge(main_df, df_global, on='Date', how='left')
        
        # Sort dataset by Date descending (Newest first)
        final_dataset = final_dataset.sort_values('Date', ascending=False)
        
        # Persist data to CSV
        output_name = "final_gold_dataset.csv"
        final_dataset.to_csv(output_name, index=False)
        print(f"\n✅ DONE! File saved: {output_name}")
        print(f"Total Rows: {len(final_dataset)}")
        print(final_dataset.head())
    else:
        print("\n❌ Failed to create complete dataset.")
