import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Data Paths
historical_prices_path = "../data/HistoricalPrices.csv"
all_stocks_data_path = "../data/all_stocks_5yr.csv"
train_historical_path = "../data/train_historical.csv"
test_historical_path = "../data/test_historical.csv"
train_stocks_with_features_path = "../data/train_stocks_with_features.csv"
test_stocks_with_features_path = "../data/test_stocks_with_features.csv"

# Function for loading data with pre-processing
def load_and_preprocess_data(file_path, date_column, rename_columns=None):
    """
    Loads data, converts dates, converts column names to lowercase, and corrects errors in columns.
    """
   # Loading data
    data = pd.read_csv(file_path)

    # Convert all column names to lowercase
    data.columns = [col.strip().lower() for col in data.columns]

    # Rename columns (if necessary)
    if rename_columns:
        data.rename(columns=rename_columns, inplace=True)

    # Date conversion
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    
    # Removing rows with incorrect dates
    data = data.dropna(subset=[date_column])

    # Remove duplicates
    data = data.drop_duplicates()

    # Remove missing values
    data = data.dropna()

    return data

# Function for adding technical indicators
def add_indicators(data):
    """
    Adds indicators (Bollinger Bands, RSI, MACD) to the data.
    """
    data = data.sort_values(by=["name", "date"])  # Let's make sure the data is sorted

    # Bollinger Bands
    bb_indicator = BollingerBands(close=data["close"], window=20, window_dev=2)
    data["bb_high"] = bb_indicator.bollinger_hband()
    data["bb_low"] = bb_indicator.bollinger_lband()

    # RSI
    rsi_indicator = RSIIndicator(close=data["close"], window=14)
    data["rsi"] = rsi_indicator.rsi()

    # MACD
    macd_indicator = MACD(close=data["close"], window_slow=26, window_fast=12, window_sign=9)
    data["macd"] = macd_indicator.macd()
    data["macd_signal"] = macd_indicator.macd_signal()

    return data

# Function to add target variable
def add_target(data):
    """
    Adds a target variable.
    """
    data["return_d1_d2"] = data["close"].shift(-2) - data["close"].shift(-1)
    data["target"] = data["return_d1_d2"].apply(lambda x: 1 if x > 0 else -1)
    return data

# Basic data processing function
def features_engineering():
    # Processing historical data
    historical_data = load_and_preprocess_data(
        historical_prices_path,
        date_column="date"
    )

    # Split data into train/test
    train_historical = historical_data[historical_data["date"] < "2017-01-01"]
    test_historical = historical_data[historical_data["date"] >= "2017-01-01"]

    # Saving processed historical data
    train_historical.to_csv(train_historical_path, index=False)
    test_historical.to_csv(test_historical_path, index=False)
    print("Historical data processed and saved.")

    # Processing stock data
    stocks_data = load_and_preprocess_data(
        all_stocks_data_path,
        date_column="date"
    )

    # Split data into train/test
    train_stocks = stocks_data[stocks_data["date"] < "2017-01-01"]
    test_stocks = stocks_data[stocks_data["date"] >= "2017-01-01"]

    # Adding indicators
    train_stocks = add_indicators(train_stocks)
    test_stocks = add_indicators(test_stocks)

    # Adding a target variable
    train_stocks = add_target(train_stocks)
    test_stocks = add_target(test_stocks)

    # Removing lines with NaN (caused by indicators)
    train_stocks = train_stocks.dropna()
    test_stocks = test_stocks.dropna()

    # Saving processed data
    train_stocks.to_csv(train_stocks_with_features_path, index=False)
    test_stocks.to_csv(test_stocks_with_features_path, index=False)
    print("Stock data processed and saved.")

# Code execution
if __name__ == "__main__":
    features_engineering()
