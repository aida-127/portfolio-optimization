# scripts/task2_forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 2: TIME SERIES FORECASTING - SIMPLE VERSION")
print("=" * 70)

# Step 1: Load Task 1 data
print("\n1. LOADING DATA")
print("-" * 40)

try:
    data = pd.read_csv('data/processed/combined_financial_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    print(f"✓ Data loaded: {len(data)} records")
    print(f"✓ Assets: {data['Ticker'].unique().tolist()}")
except:
    print("✗ Error loading data. Run Task 1 first!")
    exit()

# Step 2: Prepare TSLA data for forecasting
print("\n2. PREPARING TSLA DATA")
print("-" * 40)

tsla = data[data['Ticker'] == 'TSLA'].copy()
tsla = tsla.sort_values('Date')
tsla = tsla.set_index('Date')

# Use only 80% for training, 20% for testing
split_idx = int(len(tsla) * 0.8)
train = tsla.iloc[:split_idx]
test = tsla.iloc[split_idx:]

print(f"✓ TSLA records: {len(tsla)}")
print(f"✓ Training: {len(train)} records ({train.index.min().date()} to {train.index.max().date()})")
print(f"✓ Testing:  {len(test)} records ({test.index.min().date()} to {test.index.max().date()})")

# Step 3: Simple Moving Average Model
print("\n3. SIMPLE MOVING AVERAGE FORECAST")
print("-" * 40)

# Use 30-day moving average to predict next day
window = 30
predictions = []

for i in range(len(test)):
    if i < window:
        # Use last 'window' days from training + previous test days
        last_values = pd.concat([train['Close'][-window+i:], test['Close'][:i]])[-window:]
    else:
        # Use last 'window' predictions
        last_values = pd.Series(predictions[-window:])
    
    pred = last_values.mean()  # Simple average
    predictions.append(pred)

# Convert to numpy array
predictions = np.array(predictions)
actual = test['Close'].values

# Calculate errors
errors = actual - predictions
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
mape = np.mean(np.abs(errors / actual)) * 100

print(f"✓ Moving Average (30-day) Results:")
print(f"  MAE:  ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")

# Step 4: Simple Linear Trend Model
print("\n4. LINEAR TREND FORECAST")
print("-" * 40)

# Fit a simple linear trend to training data
train_dates = np.arange(len(train))
train_prices = train['Close'].values

# Linear regression: y = mx + b
m, b = np.polyfit(train_dates, train_prices, 1)

# Predict for test period
test_dates = np.arange(len(train), len(train) + len(test))
linear_predictions = m * test_dates + b

# Calculate errors for linear model
linear_errors = actual - linear_predictions
linear_mae = np.mean(np.abs(linear_errors))
linear_rmse = np.sqrt(np.mean(linear_errors ** 2))
linear_mape = np.mean(np.abs(linear_errors / actual)) * 100

print(f"✓ Linear Trend Results:")
print(f"  MAE:  ${linear_mae:.2f}")
print(f"  RMSE: ${linear_rmse:.2f}")
print(f"  MAPE: {linear_mape:.2f}%")

# Step 5: Compare models
print("\n5. MODEL COMPARISON")
print("-" * 40)

comparison = pd.DataFrame({
    'Model': ['Moving Average (30-day)', 'Linear Trend'],
    'MAE': [mae, linear_mae],
    'RMSE': [rmse, linear_rmse],
    'MAPE': [mape, linear_mape]
})

print(comparison.to_string(index=False))

# Determine best model
best_model_idx = comparison['MAE'].idxmin()
best_model = comparison.loc[best_model_idx, 'Model']
print(f"\n✓ Best model: {best_model} (lowest MAE)")

# Step 6: Visualize results
print("\n6. CREATING VISUALIZATIONS")
print("-" * 40)

plt.figure(figsize=(15, 10))

# Plot 1: Full timeline
plt.subplot(2, 2, 1)
plt.plot(train.index, train['Close'], label='Training Data', color='blue', linewidth=2)
plt.plot(test.index, actual, label='Actual Test', color='green', linewidth=2)
plt.plot(test.index, predictions, label='Moving Average Pred', color='red', linestyle='--')
plt.plot(test.index, linear_predictions, label='Linear Trend Pred', color='orange', linestyle='--')
plt.title('TSLA: Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Zoom on test period
plt.subplot(2, 2, 2)
plt.plot(test.index, actual, label='Actual', color='green', linewidth=2, marker='o', markersize=3)
plt.plot(test.index, predictions, label='Moving Average', color='red', linestyle='--', marker='s', markersize=3)
plt.plot(test.index, linear_predictions, label='Linear Trend', color='orange', linestyle='--', marker='^', markersize=3)
plt.title('Test Period: Detailed View')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Errors distribution
plt.subplot(2, 2, 3)
plt.hist(errors, bins=50, alpha=0.7, label=f'Moving Avg (MAE: ${mae:.2f})', color='red')
plt.hist(linear_errors, bins=50, alpha=0.7, label=f'Linear Trend (MAE: ${linear_mae:.2f})', color='orange')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Model comparison
plt.subplot(2, 2, 4)
x = np.arange(2)
width = 0.25

plt.bar(x - width, [mae, linear_mae], width, label='MAE', color='skyblue')
plt.bar(x, [rmse, linear_rmse], width, label='RMSE', color='lightcoral')
plt.bar(x + width, [mape, linear_mape], width, label='MAPE (%)', color='lightgreen')

plt.xlabel('Model')
plt.ylabel('Error Metric')
plt.title('Model Performance Comparison')
plt.xticks(x, ['Moving Avg', 'Linear Trend'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data/processed/task2_forecasting_results.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: data/processed/task2_forecasting_results.png")
plt.show()

# Step 7: Save predictions
print("\n7. SAVING RESULTS")
print("-" * 40)

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'Date': test.index,
    'Actual_Price': actual,
    'MovingAvg_Prediction': predictions,
    'LinearTrend_Prediction': linear_predictions,
    'MovingAvg_Error': errors,
    'LinearTrend_Error': linear_errors
})

predictions_df.to_csv('data/processed/task2_predictions.csv', index=False)
print(f"✓ Predictions saved: data/processed/task2_predictions.csv")

# Save summary
summary = f"""
TASK 2 SUMMARY - TIME SERIES FORECASTING
=========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA:
.
Training period: {train.index.min().date()} to {train.index.max().date()}
Testing period:  {test.index.min().date()} to {test.index.max().date()}
Training samples: {len(train)}
Testing samples:  {len(test)}

MODEL PERFORMANCE:
1. Moving Average (30-day):
   - MAE:  ${mae:.2f}
   - RMSE: ${rmse:.2f}
   - MAPE: {mape:.2f}%

2. Linear Trend Model:
   - MAE:  ${linear_mae:.2f}
   - RMSE: ${linear_rmse:.2f}
   - MAPE: {linear_mape:.2f}%

BEST MODEL: {best_model}
"""

with open('data/processed/task2_summary.txt', 'w') as f:
    f.write(summary)

print(f"✓ Summary saved: data/processed/task2_summary.txt")

print("\n" + "="*70)
print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nNext: Task 3 - Forecast Future Market Trends")