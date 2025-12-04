from curl_cffi import requests
import yfinance as yf
import pandas as pd
import time
from datetime import datetime


def fetch_tgju_data(slug, name_in_df):
    print(f"Fetching {slug} ({name_in_df}) from TGJU ---")
    base_url = "https://www.tgju.org"
    history_url = f"{base_url}/profile/{slug}/history"
    ajax_url = f"{base_url}/profile/{slug}/history/ajax"

    session = requests.Session(impersonate="chrome120")
    headers = {
        'Authority': 'www.tgju.org',
        'X-Requested-With': 'XMLHttpRequest',
        'User-Agent': 'Mozila/5.0 (Window NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        session.get(history_url, headers=headers)
        params = {
            "draw": "1", "start": "0", "length": "10000",
            "search[regex]": "false", "_:": str(int(time.time()))
        }
        response = session.get(ajax_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()['data']

            clean_rows = []
            for row in data:
                try:
                    close_price = float(str(row[3]).replace(',', ''))
                    date_str = row[6]

                    clean_rows.append({
                        'Date': date_str,
                        name_in_df: close_price
                    })
                except:
                    continue

            df = pd.DataFrame(clean_rows)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"Successfully fetched {len(df)} rows for {name_in_df}")
            return df
        else:
            print(f"Failed to fetch {slug}. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {slug}: {e}")



def fetch_global_data():
    print("\n--- Fetching Global Data (Gold Ounce & Brent Oil) ---")

    tickers = ['GC=F', 'BZ=F'] 

    df = yf.download(tickers, period="max", interval="1d")

    df = df['Close'].reset_index()

    df.rename(columns={'Date': 'Date', 'GC=F': 'Ounce', 'BZ=F': 'Oil'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    print(f"Fetched global data. Rows: {len(df)}")
    return df



df_gold = fetch_tgju_data("gerams18", "Gold_IRR")

df_usd = fetch_tgju_data("price_dollar_rl", "USD_IRR")

df_global = fetch_global_data()

if df_gold is not None and df_usd is not None and df_global is not None:
    print("\n--- Merging Data ---")

    final_df = pd.merge(df_gold, df_usd, on='Date', how='inner')

    final_df = pd.merge(final_df, df_global, on='Date', how='left')

    final_df.fillna(method='ffill', inplace=True)

    final_df.sort_values('Date', inplace=True)

    final_df.to_csv("final_dataset.csv", index=False)
    print("\nMission Complete!")
    print(final_df.head())
    print(final_df.tail())
    print(f"\nSaved to 'final_dataset.csv' with {len(final_df)} rows.")
else:
    print("Some data sources failed. Please check connections.")