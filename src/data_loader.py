import pandas as pd
import yfinance as yf
from curl_cffi import requests
import re
import os
import sys
from datetime import datetime

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..config.settings import Config
from .utils import logger

class DataLoader:
    def __init__(self):
        self.data_raw_dir = Config.DATA_RAW_DIR

    @staticmethod
    def clean_value(x):
        """
        Clean numeric values and remove HTML tags/artifacts.
        """
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            # Remove HTML tags
            x = re.sub(r'<[^>]+>', '', x)
            # Remove commas and percentage signs
            x = x.replace(',', '').replace('%', '')
            try:
                return float(x)
            except ValueError:
                return 0.0
        return 0.0

    def get_tgju_history(self, item_slug):
        """
        Fetch historical data from TGJU using curl_cffi to bypass WAF/Cloudflare.
        """
        url = f"https://api.tgju.org/v1/market/indicator/summary-table-data/{item_slug}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': f'https://www.tgju.org/profile/{item_slug}/history',
            'Origin': 'https://www.tgju.org',
            'Accept': 'application/json, text/plain, */*',
        }

        logger.info(f"Fetching {item_slug} data from TGJU...")

        try:
            # Use impersonate to mimic a real browser and bypass firewall
            session = requests.Session()
            response = session.get(url, headers=headers, impersonate="chrome110")

            if response.status_code == 200:
                json_data = response.json()
                if 'data' in json_data:
                    df = pd.DataFrame(json_data['data'])
                    
                    # Map columns dynamically based on response structure
                    cols = ['Open', 'Low', 'High', 'Close', 'Change', 'Percent', 'Date_G', 'Date_J']
                    current_cols = len(df.columns)
                    if current_cols >= 8:
                        df.columns = cols + [f'col_{i}' for i in range(8, current_cols)]
                    else:
                        df.columns = cols[:current_cols]
                    
                    # Select necessary columns
                    final_df = df[['Date_G', 'Close']].copy()
                    final_df['Close'] = final_df['Close'].apply(self.clean_value)
                    
                    # Standardize Date format
                    final_df['Date'] = pd.to_datetime(final_df['Date_G']).dt.date
                    final_df = final_df.drop(columns=['Date_G'])
                    
                    # Rename price column based on item type
                    col_name = 'Gold_IRR' if 'geram' in item_slug else 'USD_IRR'
                    final_df = final_df.rename(columns={'Close': col_name})
                    
                    logger.info(f"✅ Fetched {item_slug}: {len(final_df)} rows.")
                    return final_df
                else:
                    logger.error(f"❌ Key 'data' not found in API response for {item_slug}")
                    return None
            else:
                logger.error(f"❌ Error fetching {item_slug}: Status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"❌ Exception while fetching {item_slug}: {e}")
            return None

    def fetch_global_data(self):
        """
        Fetch global market data (Gold Ounce, Crude Oil) from Yahoo Finance.
        """
        logger.info("Fetching global data from Yahoo Finance...")
        try:
            # Global Gold (Ounce)
            gold = yf.Ticker("GC=F")
            df_gold = gold.history(period="5y")[['Close']].rename(columns={'Close': 'Ounce_USD'}).reset_index()
            
            # Crude Oil (Brent)
            oil = yf.Ticker("BZ=F")
            df_oil = oil.history(period="5y")[['Close']].rename(columns={'Close': 'Oil_USD'}).reset_index()
            
            # Normalize Date column (remove timezone info if present)
            df_gold['Date'] = pd.to_datetime(df_gold['Date']).dt.date
            df_oil['Date'] = pd.to_datetime(df_oil['Date']).dt.date
            
            # Merge global datasets
            df_global = pd.merge(df_gold, df_oil, on='Date', how='outer')
            logger.info(f"✅ Fetched global data: {len(df_global)} rows.")
            return df_global
        except Exception as e:
            logger.error(f"❌ Yahoo Finance Error: {e}")
            return None

    def fetch_data(self):
        """
        Main Pipeline: Download Local Data -> Download Global Data -> Merge -> Clean.
        """
        # 1. Download Local Data (Gold & USD)
        df_gold = self.get_tgju_history(Config.TICKER_GOLD)
        df_usd = self.get_tgju_history(Config.TICKER_USD)

        # 2. Merge Local Data
        main_df = pd.DataFrame()
        if df_gold is not None and df_usd is not None:
            main_df = pd.merge(df_gold, df_usd, on='Date', how='outer')
        elif df_gold is not None:
            main_df = df_gold
        elif df_usd is not None:
            main_df = df_usd
        
        if main_df.empty:
            logger.critical("Failed to fetch local data!")
            return None

        # 3. Download Global Data
        df_global = self.fetch_global_data()

        # 4. Final Merge
        logger.info("Merging local and global datasets...")
        if df_global is not None:
            final_dataset = pd.merge(main_df, df_global, on='Date', how='left')
        else:
            final_dataset = main_df

        # 5. Post-Processing
        # Convert Date to Index and Sort Ascending (Past -> Future) for Time Series
        final_dataset['Date'] = pd.to_datetime(final_dataset['Date'])
        final_dataset.set_index('Date', inplace=True)
        final_dataset.sort_index(ascending=True, inplace=True)
        
        # Handle Missing Values (Forward Fill is standard for financial time series)
        final_dataset.ffill(inplace=True)
        final_dataset.dropna(inplace=True)

        logger.info(f"✅ Final dataset ready. Shape: {final_dataset.shape}")
        return final_dataset

    def save_raw_data(self, df, filename="final_gold_dataset.csv"):
        """Save the raw merged dataset to disk."""
        path = os.path.join(self.data_raw_dir, filename)
        df.to_csv(path)
        logger.info(f"Raw data saved to {path}")

    def load_raw_data(self, filename="final_gold_dataset.csv"):
        """Load raw dataset from disk."""
        path = os.path.join(self.data_raw_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='Date', parse_dates=True)
            df.sort_index(ascending=True, inplace=True)
            logger.info("Raw data loaded from local file.")
            return df
        else:
            logger.warning("Raw data file not found.")
            return None
