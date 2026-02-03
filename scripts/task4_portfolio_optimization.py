# scripts/task4_portfolio_optimization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 4: PORTFOLIO OPTIMIZATION")
print("=" * 70)

# Step 1: Load all data
print("\n1. LOADING DATA")
print("-" * 40)

try:
    # Load historical data
    data = pd.read_csv('data/processed/combined_financial_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    print(f"✓ Historical data loaded: {len(data)} records")
    print(f"✓ Assets: {data['Ticker'].unique().tolist()}")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("Make sure Task 1 has been run!")
    exit()

# Step 2: Prepare data for portfolio optimization
print("\n2. PREPARING PORTFOLIO DATA")
print("-" * 40)

# Separate assets
tsla_data = data[data['Ticker'] == 'TSLA'].sort_values('Date')
bnd_data = data[data['Ticker'] == 'BND'].sort_values('Date')
spy_data = data[data['Ticker'] == 'SPY'].sort_values('Date')

# Calculate daily returns
tsla_returns = tsla_data['Close'].pct_change().dropna()
bnd_returns = bnd_data['Close'].pct_change().dropna()
spy_returns = spy_data['Close'].pct_change().dropna()

# Align all returns to same length
min_length = min(len(tsla_returns), len(bnd_returns), len(spy_returns))
tsla_returns = tsla_returns.iloc[:min_length]
bnd_returns = bnd_returns.iloc[:min_length]
spy_returns = spy_returns.iloc[:min_length]

# Expected returns (annualized)
tsla_expected = tsla_returns.mean() * 252
bnd_expected = bnd_returns.mean() * 252
spy_expected = spy_returns.mean() * 252

print(f"✓ Expected Annual Returns:")
print(f"  TSLA: {tsla_expected*100:.2f}%")
print(f"  BND:  {bnd_expected*100:.2f}%")
print(f"  SPY:  {spy_expected*100:.2f}%")

# Step 3: Portfolio optimization
print("\n3. PORTFOLIO OPTIMIZATION")
print("-" * 40)

# Combine returns for covariance matrix
returns_df = pd.DataFrame({
    'TSLA': tsla_returns.values,
    'BND': bnd_returns.values,
    'SPY': spy_returns.values
})

# Expected returns vector
expected_returns = np.array([tsla_expected, bnd_expected, spy_expected])

# Covariance matrix (annualized)
cov_matrix = returns_df.cov() * 252

print(f"✓ Covariance Matrix (annualized):")
print(f"  TSLA Variance: {cov_matrix.iloc[0,0]*100:.2f}%")
print(f"  BND Variance:  {cov_matrix.iloc[1,1]*100:.2f}%")
print(f"  SPY Variance:  {cov_matrix.iloc[2,2]*100:.2f}%")
print(f"  TSLA-BND Correlation: {cov_matrix.iloc[0,1]/np.sqrt(cov_matrix.iloc[0,0]*cov_matrix.iloc[1,1]):.3f}")
print(f"  TSLA-SPY Correlation: {cov_matrix.iloc[0,2]/np.sqrt(cov_matrix.iloc[0,0]*cov_matrix.iloc[2,2]):.3f}")

# Step 4: Generate random portfolios
print("\n4. GENERATING EFFICIENT FRONTIER")
print("-" * 40)

np.random.seed(42)
num_portfolios = 10000
results = np.zeros((num_portfolios, 6))  # Return, Volatility, Sharpe, TSLA, BND, SPY weights

for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(3)
    weights = weights / weights.sum()
    
    # Portfolio metrics
    port_return = np.sum(weights * expected_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return / port_volatility if port_volatility > 0 else 0
    
    results[i, :] = [port_return, port_volatility, sharpe_ratio, weights[0], weights[1], weights[2]]

print(f"✓ Generated {num_portfolios:,} random portfolios")

# Step 5: Find optimal portfolios
print("\n5. FINDING OPTIMAL PORTFOLIOS")
print("-" * 40)

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe', 'TSLA', 'BND', 'SPY'])

# Max Sharpe (Tangency) portfolio
max_sharpe_idx = results_df['Sharpe'].idxmax()
max_sharpe = results_df.loc[max_sharpe_idx]

# Min Volatility portfolio
min_vol_idx = results_df['Volatility'].idxmin()
min_vol = results_df.loc[min_vol_idx]

print(f"✓ MAXIMUM SHARPE RATIO PORTFOLIO:")
print(f"  Expected Return: {max_sharpe['Return']*100:.2f}%")
print(f"  Expected Volatility: {max_sharpe['Volatility']*100:.2f}%")
print(f"  Sharpe Ratio: {max_sharpe['Sharpe']:.3f}")
print(f"  Weights: TSLA={max_sharpe['TSLA']:.1%}, BND={max_sharpe['BND']:.1%}, SPY={max_sharpe['SPY']:.1%}")

print(f"\n✓ MINIMUM VOLATILITY PORTFOLIO:")
print(f"  Expected Return: {min_vol['Return']*100:.2f}%")
print(f"  Expected Volatility: {min_vol['Volatility']*100:.2f}%")
print(f"  Sharpe Ratio: {min_vol['Sharpe']:.3f}")
print(f"  Weights: TSLA={min_vol['TSLA']:.1%}, BND={min_vol['BND']:.1%}, SPY={min_vol['SPY']:.1%}")

# Step 6: Visualizations
print("\n6. CREATING VISUALIZATIONS")
print("-" * 40)

plt.figure(figsize=(15, 10))

# Plot 1: Efficient Frontier
plt.subplot(2, 2, 1)
plt.scatter(results_df['Volatility']*100, results_df['Return']*100, 
            c=results_df['Sharpe'], cmap='viridis', alpha=0.3, s=10)
plt.scatter(max_sharpe['Volatility']*100, max_sharpe['Return']*100, 
            marker='*', color='red', s=300, label='Max Sharpe')
plt.scatter(min_vol['Volatility']*100, min_vol['Return']*100, 
            marker='o', color='green', s=200, label='Min Volatility')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier', fontsize=14, fontweight='bold')
plt.xlabel('Volatility (%)')
plt.ylabel('Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Portfolio weights heatmap
plt.subplot(2, 2, 2)
weights_matrix = np.array([results_df['TSLA'], results_df['BND'], results_df['SPY']])
plt.imshow(weights_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Weight')
plt.title('Portfolio Weights Distribution', fontsize=14, fontweight='bold')
plt.yticks([0, 1, 2], ['TSLA', 'BND', 'SPY'])
plt.xlabel('Portfolio Index')
plt.grid(False)

# Plot 3: Risk-Return tradeoff
plt.subplot(2, 2, 3)
plt.plot(results_df['Volatility']*100, results_df['Return']*100, 'b.', alpha=0.1)
plt.plot(max_sharpe['Volatility']*100, max_sharpe['Return']*100, 'r*', markersize=15, label='Max Sharpe')
plt.plot(min_vol['Volatility']*100, min_vol['Return']*100, 'go', markersize=10, label='Min Vol')
plt.title('Risk-Return Tradeoff', fontsize=14, fontweight='bold')
plt.xlabel('Volatility (%)')
plt.ylabel('Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Recommended portfolio
plt.subplot(2, 2, 4)
assets = ['TSLA', 'BND', 'SPY']
max_sharpe_weights = [max_sharpe['TSLA'], max_sharpe['BND'], max_sharpe['SPY']]
min_vol_weights = [min_vol['TSLA'], min_vol['BND'], min_vol['SPY']]

x = np.arange(len(assets))
width = 0.35

plt.bar(x - width/2, max_sharpe_weights, width, label='Max Sharpe', color='red', alpha=0.7)
plt.bar(x + width/2, min_vol_weights, width, label='Min Volatility', color='green', alpha=0.7)

plt.title('Recommended Portfolio Weights', fontsize=14, fontweight='bold')
plt.xlabel('Asset')
plt.ylabel('Weight')
plt.xticks(x, assets)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data/processed/task4_portfolio_optimization.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: data/processed/task4_portfolio_optimization.png")
plt.show()

# Step 7: Save results
print("\n7. SAVING RESULTS")
print("-" * 40)

# Save portfolio results
portfolio_results = pd.DataFrame({
    'Portfolio_Type': ['Max_Sharpe', 'Min_Volatility'],
    'Expected_Return': [max_sharpe['Return'], min_vol['Return']],
    'Volatility': [max_sharpe['Volatility'], min_vol['Volatility']],
    'Sharpe_Ratio': [max_sharpe['Sharpe'], min_vol['Sharpe']],
    'TSLA_Weight': [max_sharpe['TSLA'], min_vol['TSLA']],
    'BND_Weight': [max_sharpe['BND'], min_vol['BND']],
    'SPY_Weight': [max_sharpe['SPY'], min_vol['SPY']]
})

portfolio_results.to_csv('data/processed/task4_portfolio_results.csv', index=False)
print(f"✓ Portfolio results saved: data/processed/task4_portfolio_results.csv")

# Save recommendation
from datetime import datetime
recommendation = f"""
TASK 4 SUMMARY - PORTFOLIO OPTIMIZATION
=========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA:
- Assets: TSLA, BND, SPY
- Historical period: {data['Date'].min().date()} to {data['Date'].max().date()}
- Portfolio simulations: {num_portfolios:,}

OPTIMAL PORTFOLIOS:

1. MAXIMUM SHARPE RATIO (TANGENCY PORTFOLIO):
   - Expected Return: {max_sharpe['Return']*100:.2f}%
   - Expected Volatility: {max_sharpe['Volatility']*100:.2f}%
   - Sharpe Ratio: {max_sharpe['Sharpe']:.3f}
   - Recommended Weights:
     * TSLA: {max_sharpe['TSLA']:.1%}
     * BND: {max_sharpe['BND']:.1%}
     * SPY: {max_sharpe['SPY']:.1%}

2. MINIMUM VOLATILITY PORTFOLIO:
   - Expected Return: {min_vol['Return']*100:.2f}%
   - Expected Volatility: {min_vol['Volatility']*100:.2f}%
   - Sharpe Ratio: {min_vol['Sharpe']:.3f}
   - Recommended Weights:
     * TSLA: {min_vol['TSLA']:.1%}
     * BND: {min_vol['BND']:.1%}
     * SPY: {min_vol['SPY']:.1%}

RECOMMENDATION:
{'Based on risk-return optimization, the MAXIMUM SHARPE PORTFOLIO is recommended ' + 
'for investors seeking optimal risk-adjusted returns.' if max_sharpe['Sharpe'] > min_vol['Sharpe'] else 
'For risk-averse investors, the MINIMUM VOLATILITY PORTFOLIO is recommended.'}

IMPLEMENTATION:
1. Allocate capital according to recommended weights
2. Rebalance portfolio quarterly
3. Monitor performance monthly
4. Adjust strategy if market conditions change
"""

with open('data/processed/task4_recommendation.txt', 'w', encoding='utf-8') as f:
    f.write(recommendation)

print(f"✓ Recommendation saved: data/processed/task4_recommendation.txt")

print("\n" + "="*70)
print("TASK 4 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nNext: Task 5 - Strategy Backtesting")