# scripts/task1_data_preprocessing.py - Updated Data Fetching Section

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fetch_financial_data():
    """Fetch financial data for TSLA, BND, and SPY"""
    print("=" * 50)
    print("FETCHING FINANCIAL DATA")
    print("=" * 50)
    
    # Define assets
    assets = {
        'TSLA': 'Tesla Inc.',
        'BND': 'Vanguard Total Bond Market ETF', 
        'SPY': 'SPDR S&P 500 ETF Trust'
    }
    
    # Date range
    start_date = '2015-01-01'
    end_date = '2026-01-15'
    
    data_frames = {}
    
    for ticker, name in assets.items():
        print(f"\nFetching {ticker} ({name})...")
        try:
            # Fetch data with adjusted close
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"  ✗ No data found for {ticker}")
                continue
            
            # Add ticker column
            data['Ticker'] = ticker
            data['Asset_Name'] = name
            
            # Reset index to have Date as a column
            data = data.reset_index()
            
            # Display info
            print(f"  ✓ Records: {len(data)}")
            print(f"  ✓ Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
            print(f"  ✓ Columns: {', '.join(data.columns)}")
            
            data_frames[ticker] = data
            
        except Exception as e:
            print(f"  ✗ Error fetching {ticker}: {str(e)}")
    
    return data_frames

def combine_and_clean_data(data_frames):
    """Combine all dataframes and clean"""
    if not data_frames:
        print("No valid data to combine.")
        return None
    
    # Combine all data
    combined_data = pd.concat(data_frames.values(), ignore_index=True)
    
    # Sort by Date and Ticker
    combined_data = combined_data.sort_values(['Date', 'Ticker'])
    
    # Reset index
    combined_data = combined_data.reset_index(drop=True)
    
    return combined_data

def calculate_returns_and_volatility(data):
    """Calculate daily returns and rolling volatility"""
    if data is None:
        return None
    
    print("\n" + "=" * 50)
    print("CALCULATING RETURNS & VOLATILITY")
    print("=" * 50)
    
    # Calculate daily returns
    data['Daily_Return'] = data.groupby('Ticker')['Adj Close'].pct_change()
    
    # Calculate rolling volatility (30-day window)
    data['Rolling_Vol_30d'] = data.groupby('Ticker')['Daily_Return'].transform(
        lambda x: x.rolling(window=30).std()
    )
    
    # Calculate cumulative returns
    data['Cumulative_Return'] = data.groupby('Ticker')['Daily_Return'].transform(
        lambda x: (1 + x).cumprod()
    )
    
    return data

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("GMF INVESTMENTS - TASK 1: DATA PREPROCESSING & EXPLORATION")
    print("=" * 70)
    
    print("\n1. FETCHING DATA")
    print("-" * 40)
    
    # Fetch data
    data_dict = fetch_financial_data()
    
    if not data_dict:
        print("\n✗ No data fetched. Exiting...")
        exit()
    
    print("\n" + "=" * 50)
    print("DATA FETCHING COMPLETE")
    print("=" * 50)
    
    # Combine data
    combined_data = combine_and_clean_data(data_dict)
    
    if combined_data is not None:
        print(f"\n✓ Combined data shape: {combined_data.shape}")
        print(f"✓ Assets: {combined_data['Ticker'].unique().tolist()}")
        
        # Calculate metrics
        combined_data = calculate_returns_and_volatility(combined_data)
        
        # Save to CSV
        output_path = "data/processed/combined_financial_data.csv"
        combined_data.to_csv(output_path, index=False)
        print(f"\n✓ Data saved to: {output_path}")
        
        # Display summary
        print("\n" + "=" * 50)
        print("DATA SUMMARY")
        print("=" * 50)
        print(f"Total records: {len(combined_data)}")
        print(f"Date range: {combined_data['Date'].min().date()} to {combined_data['Date'].max().date()}")
        
        for ticker in combined_data['Ticker'].unique():
            ticker_data = combined_data[combined_data['Ticker'] == ticker]
            print(f"\n{ticker}:")
            print(f"  Records: {len(ticker_data)}")
            print(f"  Avg Daily Return: {ticker_data['Daily_Return'].mean():.4%}")
            print(f"  Avg Volatility (30d): {ticker_data['Rolling_Vol_30d'].mean():.4%}")
    else:
        print("\n✗ No valid data to process.")