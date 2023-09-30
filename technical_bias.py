import datetime

import numpy as np
import yfinance as yf

df = yf.download(['ZN=F'], start='2023-08-05', end=datetime.date(2023, 9, 28).strftime('%Y-%m-%d'), interval='15m')
print(df)
# Define a function to calculate technical indicators
def calculate_technical_indicators(data):
    # Calculate moving averages (e.g., 50-day and 200-day)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Calculate relative strength index (RSI)
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate stochastic oscillator
    n = 14  # Lookback period
    data['Lowest_Low'] = data['Low'].rolling(window=n).min()
    data['Highest_High'] = data['High'].rolling(window=n).max()
    data['%K'] = ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])) * 100

    # Calculate Volume Moving Average (VMA) using ATR as the volatility measure
    atr_window = 14  # ATR lookback period
    data['ATR'] = data[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1)
    print(data)
    data['VMA_10'] = data['Volume'].rolling(window=10).apply(lambda x: np.sum(x) / 10, raw=True)

    # Calculate Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Std_20'] = data['Close'].rolling(window=20).std()
    data['Upper_Bollinger'] = data['SMA_20'] + 2 * data['Std_20']
    data['Lower_Bollinger'] = data['SMA_20'] - 2 * data['Std_20']

    # Calculate Keltner Channels
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Upper_Keltner'] = data['EMA_20'] + 2 * data['ATR']
    data['Lower_Keltner'] = data['EMA_20'] - 2 * data['ATR']

    # Calculate Ichimoku Cloud (for simplicity, using Tenkan-sen and Kijun-sen)
    data['Tenkan_sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
    data['Kijun_sen'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2

    # Detect Inside Bar pattern
    data['Inside_Bar'] = np.where((data['High'].shift(1) > data['High']) & (data['Low'].shift(1) < data['Low']), 1, 0)

    # Detect Hammer pattern (assuming a lower shadow at least twice the size of the body)
    data['Hammer'] = np.where((data['Low'] - data['Open'] >= 2 * (data['Close'] - data['Open']))
                              & (data['High'] - data['Close'] <= 0.1 * (data['Close'] - data['Open'])), 1, 0)

    # Detect Wedge pattern (simplified as a narrowing range)
    data['Wedge'] = np.where((data['High'].rolling(window=5).max() - data['Low'].rolling(window=5).min()) /
                             (data['High'] - data['Low']) <= 0.25, 1, 0)

    return data

# Calculate technical indicators
df = calculate_technical_indicators(df)

# Define the consensus indicator bias
def consensus_bias(data):
    bullish_count = 0
    bearish_count = 0

    # Define conditions for each indicator
    sma_50_condition = data['Close'] > data['SMA_50']
    sma_200_condition = data['Close'] > data['SMA_200']
    ema_50_condition = data['Close'] > data['EMA_50']
    ema_200_condition = data['Close'] > data['EMA_200']
    rsi_condition = data['RSI'] > 50
    stochastic_condition = data['%K'] > 50
    vma_condition = data['Volume'] > data['VMA_10']
    upper_bollinger_condition = data['Close'] > data['Upper_Bollinger']
    lower_bollinger_condition = data['Close'] < data['Lower_Bollinger']
    upper_keltner_condition = data['Close'] > data['Upper_Keltner']
    lower_keltner_condition = data['Close'] < data['Lower_Keltner']
    tenkan_kijun_condition = data['Tenkan_sen'] > data['Kijun_sen']
    inside_bar_condition = data['Inside_Bar'] == 1
    hammer_condition = data['Hammer'] == 1
    wedge_condition = data['Wedge'] == 1

    # Count bullish and bearish signals
    conditions = [sma_50_condition, sma_200_condition, ema_50_condition, ema_200_condition,
                  rsi_condition, stochastic_condition, vma_condition, upper_bollinger_condition,
                  lower_bollinger_condition, upper_keltner_condition, lower_keltner_condition,
                  tenkan_kijun_condition, inside_bar_condition, hammer_condition, wedge_condition]

    for condition in conditions:
        if condition.all():
            bullish_count += 1
        elif not condition.any():
            bearish_count += 1

    # Determine the consensus bias
    if bullish_count > bearish_count:
        consensus = "Bullish"
    elif bearish_count > bullish_count:
        consensus = "Bearish"
    else:
        consensus = "Neutral"

    return consensus

# Calculate the consensus indicator bias
consensus = consensus_bias(df)

# Calculate the maximum and minimum levels of each indicator
max_levels = df[['SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', '%K', 'VMA_10', 'Upper_Bollinger', 'Lower_Bollinger',
                 'Upper_Keltner', 'Lower_Keltner', 'Tenkan_sen', 'Kijun_sen', 'Inside_Bar', 'Hammer', 'Wedge']].max()
min_levels = df[['SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', '%K', 'VMA_10', 'Upper_Bollinger', 'Lower_Bollinger',
                 'Upper_Keltner', 'Lower_Keltner', 'Tenkan_sen', 'Kijun_sen', 'Inside_Bar', 'Hammer', 'Wedge']].min()

print("Consensus Indicator Bias:", consensus)
print("Maximum Levels of Indicators:")
print(max_levels)
print("Minimum Levels of Indicators:")
print(min_levels)
