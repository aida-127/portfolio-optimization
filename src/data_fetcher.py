"""
Module for fetching financial data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataFetcher:
    """Fetch financial data for portfolio optimization."""
    
    def __init__(self):
        self.assets = {
            'TSLA': 'Tesla Inc.',
            'BND': 'Vanguard Total Bond Market ETF', 
            'SPY': 'SPDR S&P 500 ETF Trust'
        }
    
    def fetch_data(self, start_date='2015-01-01', end_date='2026-01-15'):
        """
        Fetch data for all assets.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        dict
            Dictionary with ticker as key and DataFrame as value
        """
        data = {}
        
        print("=" * 50)
        print("FETCHING FINANCIAL DATA")
        print("=" * 50)
        
        for ticker, description in self.assets.items():
            print(f"\nFetching {ticker} ({description})...")
            try:
                # Fetch data
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                # Add metadata
                df['Ticker'] = ticker
                df['Description'] = description
                
                # Store in dictionary
                data[ticker] = df
                
                print(f"  ✓ Records: {len(df)}")
                print(f"  ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
                print(f"  ✓ Columns: {', '.join(df.columns.tolist()[:6])}")
                
            except Exception as e:
                print(f"  ✗ Error fetching {ticker}: {e}")
                data[ticker] = None
                
        print("\n" + "=" * 50)
        print("DATA FETCHING COMPLETE")
        print("=" * 50)
        
        return data
    
    def combine_data(self, data):
        """Combine all asset data into a single DataFrame."""
        if not data:
            print("No data to combine.")
            return None
            
        combined_list = []
        for ticker, df in data.items():
            if df is not None:
                df_reset = df.reset_index()
                combined_list.append(df_reset)
        
        if combined_list:
            combined_df = pd.concat(combined_list, ignore_index=True)
            print(f"\nCombined data: {len(combined_df)} total records")
            return combined_df
        else:
            print("No valid data to combine.")
            return None
    
    def save_data(self, data, output_path='data/processed/portfolio_data.csv'):
        """Save fetched data to CSV."""
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Combine and save
        combined_df = self.combine_data(data)
        if combined_df is not None:
            combined_df.to_csv(output_path, index=False)
            print(f"\n✓ Data saved to {output_path}")
            return combined_df
        else:
            print("\n✗ No data to save.")
            return None