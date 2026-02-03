# scripts/task3_trend_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 3: FORECAST FUTURE MARKET TRENDS")
print("=" * 70)

# Step 1: Load Task 2 predictions
print("\n1. LOADING PREVIOUS PREDICTIONS")
print("-" * 40)

try:
    # Load Task 2 predictions
    predictions = pd.read_csv('data/processed/task2_predictions.csv')
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    
    # Load original data for context
    data = pd.read_csv('data/processed/combined_financial_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    tsla_data = data[data['Ticker'] == 'TSLA'].copy()
    tsla_data = tsla_data.sort_values('Date')
    
    print(f"✓ Predictions loaded: {len(predictions)} future points")
    print(f"✓ Historical data: {len(tsla_data)} records")
    print(f"✓ Date range: {tsla_data['Date'].min().date()} to {tsla_data['Date'].max().date()}")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("Make sure Task 2 has been run first!")
    exit()

# Step 2: Generate 6-month future forecast
print("\n2. GENERATING 6-MONTH FUTURE FORECAST")
print("-" * 40)

# Use the best model from Task 2 (Moving Average)
last_price = predictions['Actual_Price'].iloc[-1]
last_pred = predictions['MovingAvg_Prediction'].iloc[-1]

# Create future dates (6 months = ~126 trading days)
last_date = predictions['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 127)]
future_dates = [d for d in future_dates if d.weekday() < 5]  # Business days only

# Simple forecast: Continue with similar pattern
# Use average error from last predictions to estimate uncertainty
avg_error = predictions['MovingAvg_Error'].abs().mean()
std_error = predictions['MovingAvg_Error'].std()

print(f"✓ Last actual price: ${last_price:.2f}")
print(f"✓ Last prediction: ${last_pred:.2f}")
print(f"✓ Average prediction error: ${avg_error:.2f}")
print(f"✓ Standard deviation of errors: ${std_error:.2f}")
print(f"✓ Forecasting {len(future_dates)} trading days (approx 6 months)")

# Generate forecast with confidence intervals
forecast_prices = []
confidence_lower = []
confidence_upper = []

# Simple trend: slight upward based on historical average return
historical_returns = tsla_data['Close'].pct_change().mean()
daily_trend = 1 + historical_returns  # Small daily growth

current_price = last_pred
for i in range(len(future_dates)):
    # Add small random component + trend
    random_component = np.random.normal(0, std_error * 0.1)
    current_price = current_price * daily_trend + random_component
    
    forecast_prices.append(current_price)
    
    # 95% confidence interval (approx ±2 standard deviations)
    ci_range = 2 * std_error * np.sqrt(i/len(future_dates) + 1)
    confidence_lower.append(current_price - ci_range)
    confidence_upper.append(current_price + ci_range)

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast_Price': forecast_prices,
    'Confidence_Lower': confidence_lower,
    'Confidence_Upper': confidence_upper,
    'Forecast_Horizon_Days': range(1, len(future_dates) + 1)
})

print(f"\n✓ Forecast generated:")
print(f"  Start: ${forecast_prices[0]:.2f}")
print(f"  End: ${forecast_prices[-1]:.2f}")
print(f"  Total change: {((forecast_prices[-1]/forecast_prices[0]-1)*100):.1f}%")

# Step 3: Analyze trends and opportunities
print("\n3. TREND ANALYSIS & MARKET OPPORTUNITIES")
print("-" * 40)

# Calculate trend metrics
price_change = forecast_prices[-1] - forecast_prices[0]
percent_change = (price_change / forecast_prices[0]) * 100

# Volatility in forecast
forecast_returns = pd.Series(forecast_prices).pct_change().dropna()
forecast_volatility = forecast_returns.std() * np.sqrt(252) * 100  # Annualized %

print("📈 TREND ANALYSIS:")
print(f"  Overall Trend: {'UPWARD' if price_change > 0 else 'DOWNWARD'}")
print(f"  Expected Price Change: ${price_change:.2f} ({percent_change:.1f}%)")
print(f"  Forecast Volatility: {forecast_volatility:.1f}% (annualized)")
print(f"  Confidence Interval Width: ±${(confidence_upper[0] - confidence_lower[0])/2:.2f} to ±${(confidence_upper[-1] - confidence_lower[-1])/2:.2f}")

print("\n💼 MARKET OPPORTUNITIES:")
if price_change > 0:
    print("  ✓ Potential buying opportunity expected")
    print(f"  ✓ Target price in 6 months: ${forecast_prices[-1]:.2f}")
else:
    print("  ⚠ Caution: Potential price decline expected")
    print(f"  ⚠ Consider reducing exposure")

print("\n⚠ RISK ASSESSMENT:")
print(f"  Confidence interval expands over time (from ±${(confidence_upper[0]-confidence_lower[0])/2:.2f} to ±${(confidence_upper[-1]-confidence_lower[-1])/2:.2f})")
print("  Long-term forecasts are less reliable")
print("  Monitor actual performance vs forecast monthly")

# Step 4: Create visualization
print("\n4. CREATING FORECAST VISUALIZATION")
print("-" * 40)

plt.figure(figsize=(15, 10))

# Plot 1: Full timeline with forecast
plt.subplot(2, 2, 1)

# Historical data
plt.plot(tsla_data['Date'], tsla_data['Close'], label='Historical Data', color='blue', alpha=0.7, linewidth=2)

# Test period predictions
plt.plot(predictions['Date'], predictions['Actual_Price'], label='Actual (Test Period)', color='green', linewidth=2)
plt.plot(predictions['Date'], predictions['MovingAvg_Prediction'], label='Model Predictions', color='red', linestyle='--', linewidth=2)

# Future forecast with confidence interval
plt.plot(forecast_df['Date'], forecast_df['Forecast_Price'], label='6-Month Forecast', color='orange', linewidth=3)
plt.fill_between(forecast_df['Date'], 
                 forecast_df['Confidence_Lower'], 
                 forecast_df['Confidence_Upper'], 
                 color='orange', alpha=0.2, label='95% Confidence Interval')

plt.title('TSLA: Historical Data & 6-Month Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Forecast zoom
plt.subplot(2, 2, 2)
plt.plot(forecast_df['Date'], forecast_df['Forecast_Price'], label='Forecast', color='orange', linewidth=3)
plt.fill_between(forecast_df['Date'], 
                 forecast_df['Confidence_Lower'], 
                 forecast_df['Confidence_Upper'], 
                 color='orange', alpha=0.2, label='95% CI')

plt.title('6-Month Forecast Detail', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Confidence interval expansion
plt.subplot(2, 2, 3)
ci_width = forecast_df['Confidence_Upper'] - forecast_df['Confidence_Lower']
plt.plot(forecast_df['Forecast_Horizon_Days'], ci_width, color='purple', linewidth=2)
plt.title('Confidence Interval Expansion Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Days into Future')
plt.ylabel('Confidence Interval Width ($)')
plt.grid(True, alpha=0.3)

# Plot 4: Opportunity/Risk assessment
plt.subplot(2, 2, 4)
categories = ['Expected Return', 'Forecast Volatility', 'CI Expansion']
values = [percent_change, forecast_volatility, (ci_width.iloc[-1]/ci_width.iloc[0]-1)*100]

colors = ['green' if x > 0 else 'red' for x in values]
bars = plt.bar(categories, values, color=colors, alpha=0.7)

# Add value labels
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value > 0 else -2),
             f'{value:.1f}%', ha='center', va='bottom' if value > 0 else 'top')

plt.title('Risk-Return Assessment', fontsize=14, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('data/processed/task3_forecast_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: data/processed/task3_forecast_analysis.png")
plt.show()

# Step 5: Save forecast results
print("\n5. SAVING FORECAST RESULTS")
print("-" * 40)

# Save forecast data
forecast_df.to_csv('data/processed/task3_forecast_data.csv', index=False)
print(f"✓ Forecast data saved: data/processed/task3_forecast_data.csv")

# Save analysis summary
summary = f"""
TASK 3 SUMMARY - FUTURE MARKET TRENDS FORECAST
================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FORECAST PARAMETERS:
- Forecast Horizon: 6 months ({len(future_dates)} trading days)
- Model Used: Moving Average (30-day) - Best from Task 2
- Last Known Price: ${last_price:.2f}
- Last Prediction: ${last_pred:.2f}

FORECAST RESULTS:
- Starting Forecast Price: ${forecast_prices[0]:.2f}
- Ending Forecast Price: ${forecast_prices[-1]:.2f}
- Expected Price Change: ${price_change:.2f} ({percent_change:.1f}%)
- Forecast Volatility: {forecast_volatility:.1f}% (annualized)

CONFIDENCE INTERVALS:
- Initial CI Width: ±${(confidence_upper[0] - confidence_lower[0])/2:.2f}
- Final CI Width: ±${(confidence_upper[-1] - confidence_lower[-1])/2:.2f}
- CI Expansion: {((confidence_upper[-1]-confidence_lower[-1])/(confidence_upper[0]-confidence_lower[0])-1)*100:.1f}%

MARKET OPPORTUNITIES:
{'✅ BUYING OPPORTUNITY: Price expected to increase by ' + str(percent_change) + '% over 6 months' if price_change > 0 else '⚠ CAUTION: Price expected to decline by ' + str(abs(percent_change)) + '% over 6 months'}

RISK ASSESSMENT:
- Forecast reliability decreases over time
- Monitor actual vs predicted monthly
- Consider hedging strategies for uncertainty

RECOMMENDATIONS:
1. {'Consider accumulating position if bullish' if price_change > 0 else 'Consider reducing exposure if bearish'}
2. Set stop-loss at ${confidence_lower[0]:.2f} (lower confidence bound)
3. Take-profit target: ${forecast_prices[-1]:.2f}
4. Re-evaluate forecast monthly
"""

with open('data/processed/task3_analysis_summary.txt', 'w') as f:
    f.write(summary)

print(f"✓ Analysis summary saved: data/processed/task3_analysis_summary.txt")

print("\n" + "="*70)
print("✅ TASK 3 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nNext: Task 4 - Portfolio Optimization")