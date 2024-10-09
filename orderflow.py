import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pyarrow.parquet as pq
import statsmodels.api as sm
from hmmlearn import hmm
from collections import deque
import networkx as nx
from statsmodels.tsa.stattools import adfuller

# New imports for advanced analysis
from sklearn.cluster import KMeans
from scipy.stats import entropy
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess Level 2 data
def load_l2_data(file_path):
    df = pq.read_table(file_path).to_pandas()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# Calculate order book imbalance
def calculate_order_book_imbalance(bids, asks, levels=10):
    bid_volume = bids['size'][:levels].sum()
    ask_volume = asks['size'][:levels].sum()
    return (bid_volume - ask_volume) / (bid_volume + ask_volume)

# Calculate liquidity weighted price
def calculate_liquidity_weighted_price(bids, asks, levels=10):
    bid_prices = bids['price'][:levels]
    bid_sizes = bids['size'][:levels]
    ask_prices = asks['price'][:levels]
    ask_sizes = asks['size'][:levels]
    
    total_bid_size = bid_sizes.sum()
    total_ask_size = ask_sizes.sum()
    
    lwp_bid = (bid_prices * bid_sizes).sum() / total_bid_size
    lwp_ask = (ask_prices * ask_sizes).sum() / total_ask_size
    
    return (lwp_bid + lwp_ask) / 2

# Calculate order flow toxicity (VPIN)
def calculate_vpin(volume, buy_volume, bucket_size):
    num_buckets = len(volume) // bucket_size
    vpin_values = []
    
    for i in range(num_buckets):
        start = i * bucket_size
        end = (i + 1) * bucket_size
        bucket_volume = volume[start:end].sum()
        bucket_buy_volume = buy_volume[start:end].sum()
        vpin = abs(2 * bucket_buy_volume - bucket_volume) / bucket_volume
        vpin_values.append(vpin)
    
    return np.mean(vpin_values)

# Detect spoofing patterns
def detect_spoofing(order_book_snapshots, threshold=0.8):
    large_orders = [snapshot[snapshot['size'] > snapshot['size'].quantile(0.95)] for snapshot in order_book_snapshots]
    canceled_orders = [set(prev.index) - set(curr.index) for prev, curr in zip(large_orders[:-1], large_orders[1:])]
    
    spoofing_score = sum(len(canceled) / len(large) if len(large) > 0 else 0 
                         for canceled, large in zip(canceled_orders, large_orders[:-1])) / len(canceled_orders)
    
    return spoofing_score > threshold

# Calculate order book pressure
def calculate_order_book_pressure(bids, asks, levels=10):
    bid_pressure = (bids['size'][:levels] * (bids['price'][:levels].max() - bids['price'][:levels])).sum()
    ask_pressure = (asks['size'][:levels] * (asks['price'][:levels] - asks['price'][:levels].min())).sum()
    return bid_pressure - ask_pressure

# Identify key support and resistance levels
def identify_support_resistance(price_data, window=20):
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(price_data.reshape(-1, 1))
    levels = sorted(kmeans.cluster_centers_.flatten())
    return levels

# Calculate order flow entropy
def calculate_order_flow_entropy(volumes, num_bins=10):
    hist, _ = np.histogram(volumes, bins=num_bins)
    return entropy(hist)

# Detect regime changes using Hidden Markov Model
def detect_regime_changes(returns, n_states=3):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(returns.reshape(-1, 1))
    hidden_states = model.predict(returns.reshape(-1, 1))
    return hidden_states

# Perform order flow seasonality analysis
def analyze_seasonality(data, period=5):
    result = adfuller(data)
    if result[1] > 0.05:  # Non-stationary
        data = data.diff().dropna()
    
    model = ARIMA(data, order=(1, 0, 1), seasonal_order=(1, 1, 1, period))
    results = model.fit()
    return results

# Advanced feature engineering
def engineer_advanced_features(df, l2_data):
    # Basic order flow features
    df['delta'] = df['buy_volume'] - df['sell_volume']
    df['cumulative_delta'] = df['delta'].cumsum()
    df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['buy_volume'] + df['sell_volume'])
    
    # Level 2 features
    df['order_book_imbalance'] = l2_data.apply(lambda row: calculate_order_book_imbalance(row['bids'], row['asks']))
    df['liquidity_weighted_price'] = l2_data.apply(lambda row: calculate_liquidity_weighted_price(row['bids'], row['asks']))
    df['vpin'] = calculate_vpin(df['volume'], df['buy_volume'], bucket_size=50)
    df['order_book_pressure'] = l2_data.apply(lambda row: calculate_order_book_pressure(row['bids'], row['asks']))
    
    # Spoofing detection
    order_book_snapshots = [row['bids'].append(row['asks']) for _, row in l2_data.iterrows()]
    df['spoofing_detected'] = detect_spoofing(order_book_snapshots)
    
    # Support and resistance
    support_resistance_levels = identify_support_resistance(df['close'].values)
    df['distance_to_support'] = df['close'] - min(level for level in support_resistance_levels if level < df['close'])
    df['distance_to_resistance'] = min(level for level in support_resistance_levels if level > df['close']) - df['close']
    
    # Order flow entropy
    df['order_flow_entropy'] = calculate_order_flow_entropy(df['volume'])
    
    # Regime detection
    df['regime'] = detect_regime_changes(df['returns'].values)
    
    # Seasonality
    seasonality_model = analyze_seasonality(df['volume'])
    df['seasonality_residuals'] = df['volume'] - seasonality_model.fittedvalues
    
    return df

# Machine Learning Model for Signal Generation
def train_advanced_ml_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model

# Generate Trading Signals
def generate_advanced_signals(df, model):
    features = ['delta', 'cumulative_delta', 'volume_imbalance', 'order_book_imbalance',
                'liquidity_weighted_price', 'vpin', 'order_book_pressure', 'spoofing_detected',
                'distance_to_support', 'distance_to_resistance', 'order_flow_entropy', 'regime',
                'seasonality_residuals']
    X = df[features]
    df['ml_signal'] = model.predict(X)
    
    # Combine ML signal with advanced metrics
    df['signal'] = 0
    df.loc[(df['ml_signal'] == 1) & (df['order_book_imbalance'] > 0.5) & (df['vpin'] < 0.3), 'signal'] = 1
    df.loc[(df['ml_signal'] == -1) & (df['order_book_imbalance'] < -0.5) & (df['vpin'] < 0.3), 'signal'] = -1
    
    return df

# Risk Management with Kelly Criterion
def apply_advanced_risk_management(df, window=252):
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    df['win_rate'] = df['strategy_returns'].rolling(window).apply(lambda x: (x > 0).mean())
    df['avg_win'] = df['strategy_returns'].rolling(window).apply(lambda x: x[x > 0].mean())
    df['avg_loss'] = df['strategy_returns'].rolling(window).apply(lambda x: x[x < 0].mean())
    
    df['kelly_fraction'] = (df['win_rate'] - ((1 - df['win_rate']) / (df['avg_win'] / -df['avg_loss']))).clip(0, 1)
    
    df['position_size'] = df['kelly_fraction'] * df['signal']
    
    return df

# Performance Evaluation
def evaluate_advanced_performance(df):
    df['strategy_returns'] = df['position_size'].shift(1) * df['returns']
    
    total_return = (1 + df['strategy_returns']).prod() - 1
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
    max_drawdown = (df['strategy_returns'].cumsum() - df['strategy_returns'].cumsum().cummax()).min()
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, (1 + df['strategy_returns']).cumprod())
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load Level 1 and Level 2 data
    df = load_and_preprocess_data('level1_data.parquet')
    l2_data = load_l2_data('level2_data.parquet')
    
    # Engineer advanced features
    df = engineer_advanced_features(df, l2_data)
    
    # Prepare data for ML model
    X = df[['delta', 'cumulative_delta', 'volume_imbalance', 'order_book_imbalance',
            'liquidity_weighted_price', 'vpin', 'order_book_pressure', 'spoofing_detected',
            'distance_to_support', 'distance_to_resistance', 'order_flow_entropy', 'regime',
            'seasonality_residuals']]
    y = np.where(df['returns'].shift(-1) > 0, 1, -1)
    
    # Train ML model
    model = train_advanced_ml_model(X, y)
    
    # Generate trading signals
    df = generate_advanced_signals(df, model)
    
    # Apply risk management
    df = apply_advanced_risk_management(df)
    
    # Evaluate performance
    evaluate_advanced_performance(df)