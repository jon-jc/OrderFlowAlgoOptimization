# Advanced Order Flow Trading Strategy with Data Conversion

This project includes Python scripts for both **converting Sierra Chart Level 1 and Level 2 data** into Parquet format and implementing an **advanced quantitative order flow trading strategy**. It is designed for traders looking to automate their data processing and trading strategies using high-quality market data from Sierra Chart’s Denali feed.

## Features

### Data Conversion
- Converts `.scid` (Level 1) and `.depth` (Level 2) data files from Sierra Chart into **Parquet format** for efficient storage and analysis.
- Automatically processes all data files in a specified directory.
- Supports order flow analysis, Level 2 market depth insights, and machine learning applications.

### Trading Strategy
- **Order Book Imbalance**: Detects the imbalance between bid and ask orders.
- **Liquidity Weighted Price**: Calculates liquidity-adjusted prices.
- **Order Flow Toxicity**: Measures the risk of informed trading based on volume imbalance.
- **Market Regime Detection**: Uses Hidden Markov Models (HMM) to detect bullish, bearish, or sideways markets.
- **Machine Learning Signals**: Employs Random Forest classifiers to generate trading signals.
- **Risk Management**: Implements dynamic position sizing using the Kelly Criterion and standard stop-loss mechanisms.
- **Level 2 Market Depth Analysis**: Analyzes multiple levels of bids and asks for robust order flow insights.

## Requirements

- Python 3.6+
- Libraries:
  - `pandas`
  - `pyarrow`
  - `sklearn`
  - `hmmlearn`
  - `numpy`

You can install the required libraries using `pip`:

```bash
pip install pandas pyarrow scikit-learn hmmlearn numpy
