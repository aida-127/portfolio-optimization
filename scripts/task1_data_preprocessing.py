#!/usr/bin/env python3
"""
Task 1: Preprocess and Explore Data
Complete implementation for Task 1 requirements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import statistical tests
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# Import project modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_fetcher import DataFetcher


class Task1Preprocessing:
    """Complete Task 1 implementation."""
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.data = None
        self.combined_df = None
        
    def run_task1(self):
        """Execute complete Task 1 workflow."""
        print("=" * 70)
        print("GMF INVESTMENTS - TASK 1: DATA PREPROCESSING & EXPLORATION")
        print("=" * 70)
        
        # 1. Fetch data
        self.fetch_and_save_data()
        
        if self.data is None:
            print("Failed to fetch data. Exiting.")
            return
        
        # 2. Clean and preprocess
        self.clean_and_preprocess()
        
        # 3. Perform EDA
        self.perform_eda()
        
        # 4. Calculate risk metrics
        self.calculate_risk_metrics()
        
        # 5. Test for stationarity
        self.test_stationarity()
        
        print("\n" + "=" * 70)
        print("TASK 1 COMPLETED SUCCESSFULLY")
        print("=" * 70)
    
    def fetch_and_save_data(self):
        """Fetch data using DataFetcher."""
        print("\n1. FETCHING DATA")
        print("-" * 40)
        
        # Fetch data
        self.data = self.fetcher.fetch_data(
            start_date='2015-01-01',
            end_date='2026-01-15'
        )
        
        # Save to CSV
        self.combined_df = self.fetcher.save_data(
            self.data,
            output_path='data/processed/portfolio_data.csv'
        )
        
        # Print summary
        print("\nData Summary:")
        print(f"Total assets: {len(self.data)}")
        for ticker, df in self.data.items():
            if df is not None:
                print(f"  {ticker}: {len(df)} records")
    
    def clean_and_preprocess(self):
        """Clean and preprocess the data."""
        print("\n\n2. DATA CLEANING & PREPROCESSING")
        print("-" * 40)
        
        # Check each asset's data
        for ticker, df in self.data.items():
            if df is not None:
                print(f"\n{ticker} - Data Cleaning:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                
                # Check for missing values
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    print(f"  Missing values found: {missing[missing > 0].to_dict()}")
                    # Forward fill missing values
                    df.ffill(inplace=True)
                    print("  ✓ Missing values filled using forward fill")
                else:
                    print("  ✓ No missing values")
                
                # Check data types
                print(f"  Data types:\n{df.dtypes}")
                
                # Add daily returns
                df['Daily_Return'] = df['Adj Close'].pct_change()
                df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
                print("  ✓ Added daily returns and log returns")
                
                # Add volatility (20-day rolling)
                df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
                print("  ✓ Added 20-day rolling volatility")
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis."""
        print("\n\n3. EXPLORATORY DATA ANALYSIS (EDA)")
        print("-" * 40)
        
        # Create EDA directory
        import os
        os.makedirs('data/processed/eda_plots', exist_ok=True)
        
        # 1. Price trends for all assets
        print("\nCreating price trend visualizations...")
        plt.figure(figsize=(15, 10))
        
        for i, (ticker, df) in enumerate(self.data.items(), 1):
            if df is not None:
                plt.subplot(3, 2, i)
                plt.plot(df.index, df['Adj Close'], linewidth=2)
                plt.title(f'{ticker} - Adjusted Close Price')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/eda_plots/price_trends.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Saved: price_trends.png")
        
        # 2. Daily returns distribution
        print("\nCreating returns distribution plots...")
        plt.figure(figsize=(15, 5))
        
        for i, (ticker, df) in enumerate(self.data.items(), 1):
            if df is not None:
                plt.subplot(1, 3, i)
                returns = df['Daily_Return'].dropna()
                plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
                plt.title(f'{ticker} - Daily Returns Distribution')
                plt.xlabel('Daily Return')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Add statistics
                mean_return = returns.mean()
                std_return = returns.std()
                plt.axvline(mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.4f}')
                plt.axvline(mean_return + std_return, color='orange', linestyle=':', label=f'±1 Std')
                plt.axvline(mean_return - std_return, color='orange', linestyle=':')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('data/processed/eda_plots/returns_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Saved: returns_distribution.png")
        
        # 3. Correlation heatmap
        print("\nCreating correlation analysis...")
        # Prepare correlation data
        correlation_data = {}
        for ticker, df in self.data.items():
            if df is not None:
                correlation_data[f'{ticker}_Return'] = df['Daily_Return']
        
        corr_df = pd.DataFrame(correlation_data).dropna()
        correlation_matrix = corr_df.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Daily Returns')
        plt.tight_layout()
        plt.savefig('data/processed/eda_plots/correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Saved: correlation_matrix.png")
        print(f"\nCorrelation Summary:")
        print(correlation_matrix.round(3))
        
        # 4. Volatility comparison
        print("\nCreating volatility comparison...")
        plt.figure(figsize=(12, 6))
        
        for ticker, df in self.data.items():
            if df is not None:
                volatility = df['Volatility_20d'].dropna()
                plt.plot(volatility.index, volatility, label=ticker, linewidth=1.5, alpha=0.8)
        
        plt.title('20-Day Rolling Volatility (Annualized)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/processed/eda_plots/volatility_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Saved: volatility_comparison.png")
    
    def calculate_risk_metrics(self):
        """Calculate key risk metrics."""
        print("\n\n4. RISK METRICS CALCULATION")
        print("-" * 40)
        
        risk_metrics = {}
        
        for ticker, df in self.data.items():
            if df is not None:
                print(f"\n{ticker} - Risk Metrics:")
                
                # Clean returns data
                returns = df['Daily_Return'].dropna()
                
                # 1. Value at Risk (VaR) - Historical method at 95% confidence
                var_95 = np.percentile(returns, 5)
                print(f"  Value at Risk (95%): {var_95:.4%}")
                
                # 2. Sharpe Ratio (assuming risk-free rate of 2%)
                risk_free_rate = 0.02 / 252  # Daily risk-free rate
                excess_returns = returns - risk_free_rate
                sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
                print(f"  Sharpe Ratio (annualized): {sharpe_ratio:.3f}")
                
                # 3. Maximum Drawdown
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                print(f"  Maximum Drawdown: {max_drawdown:.4%}")
                
                # 4. Annualized Return and Volatility
                annual_return = (1 + returns.mean()) ** 252 - 1
                annual_volatility = returns.std() * np.sqrt(252)
                print(f"  Annualized Return: {annual_return:.4%}")
                print(f"  Annualized Volatility: {annual_volatility:.4%}")
                
                # 5. Skewness and Kurtosis
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
                print(f"  Skewness: {skewness:.3f}")
                print(f"  Kurtosis: {kurtosis:.3f}")
                
                # Store metrics
                risk_metrics[ticker] = {
                    'VaR_95%': var_95,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Max_Drawdown': max_drawdown,
                    'Annual_Return': annual_return,
                    'Annual_Volatility': annual_volatility,
                    'Skewness': skewness,
                    'Kurtosis': kurtosis
                }
        
        # Create summary DataFrame
        metrics_df = pd.DataFrame(risk_metrics).T
        print("\n" + "=" * 60)
        print("RISK METRICS SUMMARY")
        print("=" * 60)
        print(metrics_df.round(4))
        
        # Save metrics to CSV
        metrics_df.to_csv('data/processed/risk_metrics.csv')
        print("\n✓ Risk metrics saved to: data/processed/risk_metrics.csv")
        
        # Visualize risk metrics
        self.visualize_risk_metrics(metrics_df)
        
        return metrics_df
    
    def visualize_risk_metrics(self, metrics_df):
        """Visualize risk metrics comparison."""
        print("\nCreating risk metrics visualization...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Sharpe Ratio comparison
        metrics_df['Sharpe_Ratio'].plot(kind='bar', ax=axes[0, 0], color='green', alpha=0.7)
        axes[0, 0].set_title('Sharpe Ratio Comparison')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Annualized Return vs Volatility
        colors = ['red', 'blue', 'green']
        for i, (ticker, row) in enumerate(metrics_df.iterrows()):
            axes[0, 1].scatter(row['Annual_Volatility'], row['Annual_Return'],
                             color=colors[i], s=200, label=ticker, alpha=0.7)
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].set_xlabel('Annual Volatility')
        axes[0, 1].set_ylabel('Annual Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Maximum Drawdown
        metrics_df['Max_Drawdown'].plot(kind='bar', ax=axes[1, 0], color='red', alpha=0.7)
        axes[1, 0].set_title('Maximum Drawdown')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Value at Risk
        metrics_df['VaR_95%'].plot(kind='bar', ax=axes[1, 1], color='orange', alpha=0.7)
        axes[1, 1].set_title('Value at Risk (95%)')
        axes[1, 1].set_ylabel('VaR (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/eda_plots/risk_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Saved: risk_metrics_comparison.png")
    
    def test_stationarity(self):
        """Test time series data for stationarity."""
        print("\n\n5. STATIONARITY TESTING")
        print("-" * 40)
        
        stationarity_results = {}
        
        for ticker, df in self.data.items():
            if df is not None:
                print(f"\n{ticker} - Stationarity Tests:")
                
                # Test 1: Original Price Series
                price_series = df['Adj Close'].dropna()
                adf_result_price = adfuller(price_series)
                
                print(f"  Price Series ADF Test:")
                print(f"    ADF Statistic: {adf_result_price[0]:.4f}")
                print(f"    p-value: {adf_result_price[1]:.4f}")
                print(f"    Critical Values:")
                for key, value in adf_result_price[4].items():
                    print(f"      {key}: {value:.4f}")
                
                # Test 2: Returns Series (should be more stationary)
                returns_series = df['Daily_Return'].dropna()
                adf_result_returns = adfuller(returns_series)
                
                print(f"\n  Returns Series ADF Test:")
                print(f"    ADF Statistic: {adf_result_returns[0]:.4f}")
                print(f"    p-value: {adf_result_returns[1]:.4f}")
                
                # Interpret results
                if adf_result_price[1] <= 0.05:
                    print(f"    ✓ Price series is STATIONARY (p <= 0.05)")
                else:
                    print(f"    ✗ Price series is NON-STATIONARY (p > 0.05)")
                
                if adf_result_returns[1] <= 0.05:
                    print(f"    ✓ Returns series is STATIONARY (p <= 0.05)")
                else:
                    print(f"    ✗ Returns series is NON-STATIONARY (p > 0.05)")
                
                # Store results
                stationarity_results[ticker] = {
                    'Price_ADF_Statistic': adf_result_price[0],
                    'Price_p_value': adf_result_price[1],
                    'Price_Stationary': adf_result_price[1] <= 0.05,
                    'Returns_ADF_Statistic': adf_result_returns[0],
                    'Returns_p_value': adf_result_returns[1],
                    'Returns_Stationary': adf_result_returns[1] <= 0.05
                }
                
                # Plot ACF for visual check
                self.plot_acf_pacf(df, ticker)
        
        # Create summary DataFrame
        stationarity_df = pd.DataFrame(stationarity_results).T
        print("\n" + "=" * 60)
        print("STATIONARITY TEST SUMMARY")
        print("=" * 60)
        print(stationarity_df.round(4))
        
        # Save results
        stationarity_df.to_csv('data/processed/stationarity_tests.csv')
        print("\n✓ Stationarity tests saved to: data/processed/stationarity_tests.csv")
        
        return stationarity_df
    
    def plot_acf_pacf(self, df, ticker):
        """Plot ACF and PACF for stationarity analysis."""
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        returns = df['Daily_Return'].dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Price series
        axes[0, 0].plot(df['Adj Close'])
        axes[0, 0].set_title(f'{ticker} - Price Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns series
        axes[0, 1].plot(returns)
        axes[0, 1].set_title(f'{ticker} - Returns Series')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF of returns
        plot_acf(returns, lags=40, ax=axes[1, 0])
        axes[1, 0].set_title(f'{ticker} - ACF of Returns')
        axes[1, 0].grid(True, alpha=0.3)
        
        # PACF of returns
        plot_pacf(returns, lags=40, ax=axes[1, 1])
        axes[1, 1].set_title(f'{ticker} - PACF of Returns')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'data/processed/eda_plots/{ticker}_stationarity.png', dpi=150, bbox_inches='tight')
        plt.show()


# Main execution
if __name__ == "__main__":
    # Create output directories
    import os
    os.makedirs('data/processed/eda_plots', exist_ok=True)
    
    # Run Task 1
    task1 = Task1Preprocessing()
    task1.run_task1()