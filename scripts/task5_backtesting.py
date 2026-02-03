# scripts/task5_backtesting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 5: STRATEGY BACKTESTING")
print("=" * 70)

# Step 1: Load data and Task 4 recommendations
print("\n1. LOADING DATA & RECOMMENDATIONS")
print("-" * 40)

try:
    # Load historical data
    data = pd.read_csv('data/processed/combined_financial_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Load Task 4 portfolio recommendations
    portfolio_recs = pd.read_csv('data/processed/task4_portfolio_results.csv')
    
    print(f"✓ Historical data: {len(data)} records")
    print(f"✓ Portfolio recommendations loaded")
    print(f"✓ Backtesting period: Last year of data")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit()

# Step 2: Define backtesting parameters
print("\n2. SETTING UP BACKTEST")
print("-" * 40)

# Use last year for backtesting (approx 252 trading days)
data = data.sort_values('Date')
last_date = data['Date'].max()
start_date = last_date - pd.Timedelta(days=365)

backtest_data = data[data['Date'] >= start_date].copy()
backtest_data = backtest_data.sort_values('Date')

print(f"✓ Backtest period: {start_date.date()} to {last_date.date()}")
print(f"✓ Trading days: {backtest_data['Date'].nunique()}")

# Step 3: Define strategies to backtest
print("\n3. DEFINING STRATEGIES")
print("-" * 40)

# Get optimal weights from Task 4
max_sharpe_row = portfolio_recs[portfolio_recs['Portfolio_Type'] == 'Max_Sharpe'].iloc[0]
min_vol_row = portfolio_recs[portfolio_recs['Portfolio_Type'] == 'Min_Volatility'].iloc[0]

# Strategy 1: Our optimized portfolio (Max Sharpe)
strategy_weights = {
    'TSLA': max_sharpe_row['TSLA_Weight'],
    'BND': max_sharpe_row['BND_Weight'],
    'SPY': max_sharpe_row['SPY_Weight']
}

# Strategy 2: Benchmark portfolio (60% SPY, 40% BND - common balanced portfolio)
benchmark_weights = {
    'TSLA': 0.0,
    'BND': 0.4,
    'SPY': 0.6
}

print(f"✓ OUR STRATEGY (Max Sharpe Portfolio):")
print(f"  TSLA: {strategy_weights['TSLA']:.1%}")
print(f"  BND:  {strategy_weights['BND']:.1%}")
print(f"  SPY:  {strategy_weights['SPY']:.1%}")

print(f"\n✓ BENCHMARK (60% SPY, 40% BND):")
print(f"  TSLA: {benchmark_weights['TSLA']:.1%}")
print(f"  BND:  {benchmark_weights['BND']:.1%}")
print(f"  SPY:  {benchmark_weights['SPY']:.1%}")

# Step 4: Run backtest simulation
print("\n4. RUNNING BACKTEST SIMULATION")
print("-" * 40)

# Prepare price data for each asset
pivot_data = backtest_data.pivot(index='Date', columns='Ticker', values='Close')
pivot_data = pivot_data[['TSLA', 'BND', 'SPY']].ffill()  # Forward fill missing values

# Calculate daily returns
returns = pivot_data.pct_change().dropna()

# Initial investment
initial_investment = 10000  # $10,000 starting capital

# Calculate portfolio returns
strategy_returns = pd.Series(0.0, index=returns.index)
benchmark_returns = pd.Series(0.0, index=returns.index)

for date in returns.index:
    # Strategy portfolio return
    strat_return = 0
    for ticker, weight in strategy_weights.items():
        if ticker in returns.columns:
            strat_return += weight * returns.loc[date, ticker]
    strategy_returns.loc[date] = strat_return
    
    # Benchmark portfolio return
    bench_return = 0
    for ticker, weight in benchmark_weights.items():
        if ticker in returns.columns:
            bench_return += weight * returns.loc[date, ticker]
    benchmark_returns.loc[date] = bench_return

print(f"✓ Simulation complete: {len(strategy_returns)} trading days")

# Step 5: Calculate performance metrics
print("\n5. CALCULATING PERFORMANCE METRICS")
print("-" * 40)

# Calculate cumulative returns
strategy_cumulative = (1 + strategy_returns).cumprod()
benchmark_cumulative = (1 + benchmark_returns).cumprod()

# Calculate final values
strategy_final = initial_investment * strategy_cumulative.iloc[-1]
benchmark_final = initial_investment * benchmark_cumulative.iloc[-1]

# Calculate total returns
strategy_total_return = (strategy_final / initial_investment - 1) * 100
benchmark_total_return = (benchmark_final / initial_investment - 1) * 100

# Annualized returns (assuming 252 trading days)
strategy_annual_return = (1 + strategy_returns.mean()) ** 252 - 1
benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1

# Sharpe ratio (assuming 0% risk-free rate for simplicity)
strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)

# Maximum drawdown
def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min() * 100

strategy_drawdown = calculate_max_drawdown(strategy_cumulative)
benchmark_drawdown = calculate_max_drawdown(benchmark_cumulative)

print(f"✓ OUR STRATEGY PERFORMANCE:")
print(f"  Final Value: ${strategy_final:,.2f}")
print(f"  Total Return: {strategy_total_return:.1f}%")
print(f"  Annualized Return: {strategy_annual_return*100:.1f}%")
print(f"  Sharpe Ratio: {strategy_sharpe:.3f}")
print(f"  Max Drawdown: {strategy_drawdown:.1f}%")

print(f"\n✓ BENCHMARK PERFORMANCE:")
print(f"  Final Value: ${benchmark_final:,.2f}")
print(f"  Total Return: {benchmark_total_return:.1f}%")
print(f"  Annualized Return: {benchmark_annual_return*100:.1f}%")
print(f"  Sharpe Ratio: {benchmark_sharpe:.3f}")
print(f"  Max Drawdown: {benchmark_drawdown:.1f}%")

# Step 6: Create visualizations
print("\n6. CREATING BACKTEST VISUALIZATIONS")
print("-" * 40)

plt.figure(figsize=(15, 10))

# Plot 1: Cumulative returns comparison
plt.subplot(2, 2, 1)
plt.plot(strategy_cumulative.index, strategy_cumulative.values, 
         label=f'Our Strategy ({strategy_total_return:.1f}%)', 
         linewidth=2, color='blue')
plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
         label=f'Benchmark ({benchmark_total_return:.1f}%)', 
         linewidth=2, color='green', linestyle='--')
plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Starting at 1.0)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Portfolio value over time ($10,000 initial)
plt.subplot(2, 2, 2)
strategy_value = initial_investment * strategy_cumulative
benchmark_value = initial_investment * benchmark_cumulative

plt.plot(strategy_value.index, strategy_value.values, 
         label=f'Our Strategy (Final: ${strategy_final:,.0f})', 
         linewidth=2, color='blue')
plt.plot(benchmark_value.index, benchmark_value.values, 
         label=f'Benchmark (Final: ${benchmark_final:,.0f})', 
         linewidth=2, color='green', linestyle='--')
plt.title('Portfolio Value Over Time ($10,000 Initial)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Performance metrics comparison
plt.subplot(2, 2, 3)
metrics = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']
strategy_metrics = [strategy_total_return, strategy_annual_return*100, strategy_sharpe, strategy_drawdown]
benchmark_metrics = [benchmark_total_return, benchmark_annual_return*100, benchmark_sharpe, benchmark_drawdown]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, strategy_metrics, width, label='Our Strategy', color='blue', alpha=0.7)
plt.bar(x + width/2, benchmark_metrics, width, label='Benchmark', color='green', alpha=0.7)

plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (s_val, b_val) in enumerate(zip(strategy_metrics, benchmark_metrics)):
    plt.text(i - width/2, s_val + (0.5 if s_val > 0 else -2), f'{s_val:.1f}', 
             ha='center', va='bottom' if s_val > 0 else 'top', fontsize=9)
    plt.text(i + width/2, b_val + (0.5 if b_val > 0 else -2), f'{b_val:.1f}', 
             ha='center', va='bottom' if b_val > 0 else 'top', fontsize=9)

# Plot 4: Monthly returns heatmap
plt.subplot(2, 2, 4)
# Resample to monthly returns
strategy_monthly = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
benchmark_monthly = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

months = strategy_monthly.index.strftime('%b %Y')
y_positions = np.arange(len(months))

plt.barh(y_positions - 0.2, strategy_monthly.values * 100, 0.4, 
         label='Our Strategy', color='blue', alpha=0.7)
plt.barh(y_positions + 0.2, benchmark_monthly.values * 100, 0.4, 
         label='Benchmark', color='green', alpha=0.7)

plt.title('Monthly Returns Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Monthly Return (%)')
plt.ylabel('Month')
plt.yticks(y_positions, months)
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('data/processed/task5_backtesting_results.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: data/processed/task5_backtesting_results.png")
plt.show()

# Step 7: Save results and conclusion
print("\n7. SAVING RESULTS & CONCLUSION")
print("-" * 40)

# Save backtest results
backtest_results = pd.DataFrame({
    'Metric': ['Final_Value', 'Total_Return', 'Annualized_Return', 'Sharpe_Ratio', 'Max_Drawdown'],
    'Our_Strategy': [strategy_final, strategy_total_return, strategy_annual_return*100, strategy_sharpe, strategy_drawdown],
    'Benchmark': [benchmark_final, benchmark_total_return, benchmark_annual_return*100, benchmark_sharpe, benchmark_drawdown]
})

backtest_results.to_csv('data/processed/task5_backtest_results.csv', index=False)
print(f"✓ Backtest results saved: data/processed/task5_backtest_results.csv")

# Save conclusion
from datetime import datetime
conclusion = f"""
TASK 5 SUMMARY - STRATEGY BACKTESTING
=========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BACKTEST SETUP:
- Backtest Period: {start_date.date()} to {last_date.date()}
- Initial Investment: ${initial_investment:,.0f}
- Our Strategy: Max Sharpe Portfolio from Task 4
- Benchmark: 60% SPY / 40% BND (balanced portfolio)

RESULTS:

OUR STRATEGY (Max Sharpe Portfolio):
- Final Portfolio Value: ${strategy_final:,.2f}
- Total Return: {strategy_total_return:.1f}%
- Annualized Return: {strategy_annual_return*100:.1f}%
- Sharpe Ratio: {strategy_sharpe:.3f}
- Maximum Drawdown: {strategy_drawdown:.1f}%

BENCHMARK (60% SPY / 40% BND):
- Final Portfolio Value: ${benchmark_final:,.2f}
- Total Return: {benchmark_total_return:.1f}%
- Annualized Return: {benchmark_annual_return*100:.1f}%
- Sharpe Ratio: {benchmark_sharpe:.3f}
- Maximum Drawdown: {benchmark_drawdown:.1f}%

PERFORMANCE COMPARISON:
- Outperformance: {strategy_total_return - benchmark_total_return:.1f}% {'(Our Strategy Wins!)' if strategy_total_return > benchmark_total_return else '(Benchmark Wins)'}
- Risk-Adjusted Outperformance: {strategy_sharpe - benchmark_sharpe:.3f} Sharpe ratio difference

CONCLUSION:
{'✅ OUR STRATEGY OUTPERFORMED THE BENCHMARK! The model-driven portfolio optimization approach delivered better returns with improved risk-adjusted performance.' if strategy_total_return > benchmark_total_return else '⚠ OUR STRATEGY UNDERPERFORMED THE BENCHMARK. The simple 60/40 portfolio proved more effective in this backtest period.'}

LIMITATIONS:
1. Backtest uses historical data - past performance doesn't guarantee future results
2. No transaction costs or taxes considered
3. Assumes perfect execution and rebalancing
4. Short backtest period (1 year)

RECOMMENDATIONS:
1. {'Continue using the optimized portfolio strategy' if strategy_total_return > benchmark_total_return else 'Consider revising the optimization approach'}
2. Monitor performance quarterly
3. Consider implementing stop-loss mechanisms
4. Regularly update the model with new data
"""

with open('data/processed/task5_conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(conclusion)

print(f"✓ Conclusion saved: data/processed/task5_conclusion.txt")

print("\n" + "="*70)
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nAll 5 Tasks Completed:")
print("1. ✅ Data Preprocessing & EDA")
print("2. ✅ Time Series Forecasting")
print("3. ✅ Future Market Trends")
print("4. ✅ Portfolio Optimization")
print("5. ✅ Strategy Backtesting")
print("\n📁 Files created in data/processed/:")
print("   task1_*, task2_*, task3_*, task4_*, task5_*")
print("\n🚀 Ready for final submission!")