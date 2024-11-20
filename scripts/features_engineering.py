import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Пути к данным
historical_prices_path = "../data/HistoricalPrices.csv"
all_stocks_data_path = "../data/all_stocks_5yr.csv"
train_historical_path = "../data/train_historical.csv"
test_historical_path = "../data/test_historical.csv"
train_stocks_with_features_path = "../data/train_stocks_with_features.csv"
test_stocks_with_features_path = "../data/test_stocks_with_features.csv"

# Функция для загрузки данных с предварительной обработкой
def load_and_preprocess_data(file_path, date_column, rename_columns=None):
    """
    Загружает данные, преобразует даты, приводит названия столбцов к нижнему регистру
    и исправляет ошибки в колонках.
    """
    # Загрузка данных
    data = pd.read_csv(file_path)

    # Приведение всех названий столбцов к нижнему регистру
    data.columns = [col.strip().lower() for col in data.columns]

    # Переименование колонок (если необходимо)
    if rename_columns:
        data.rename(columns=rename_columns, inplace=True)

    # Преобразование даты
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    
    # Удаление строк с некорректными датами
    data = data.dropna(subset=[date_column])

    # Удаление дубликатов
    data = data.drop_duplicates()

    # Удаление пропущенных значений
    data = data.dropna()

    return data

# Функция для добавления технических индикаторов
def add_indicators(data):
    """
    Добавляет индикаторы (Bollinger Bands, RSI, MACD) к данным.
    """
    data = data.sort_values(by=["name", "date"])  # Убедимся, что данные отсортированы

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

# Функция для добавления целевой переменной
def add_target(data):
    """
    Добавляет целевую переменную.
    """
    data["return_d1_d2"] = data["close"].shift(-2) - data["close"].shift(-1)
    data["target"] = data["return_d1_d2"].apply(lambda x: 1 if x > 0 else -1)
    return data

# Основная функция обработки данных
def features_engineering():
    # === Обработка исторических данных ===
    historical_data = load_and_preprocess_data(
        historical_prices_path,
        date_column="date"
    )

    # Разделение данных на train/test
    train_historical = historical_data[historical_data["date"] < "2017-01-01"]
    test_historical = historical_data[historical_data["date"] >= "2017-01-01"]

    # Сохранение обработанных исторических данных
    train_historical.to_csv(train_historical_path, index=False)
    test_historical.to_csv(test_historical_path, index=False)
    print("Historical data processed and saved.")

    # === Обработка стоковых данных ===
    stocks_data = load_and_preprocess_data(
        all_stocks_data_path,
        date_column="date"
    )

    # Разделение данных на train/test
    train_stocks = stocks_data[stocks_data["date"] < "2017-01-01"]
    test_stocks = stocks_data[stocks_data["date"] >= "2017-01-01"]

    # Добавление индикаторов
    train_stocks = add_indicators(train_stocks)
    test_stocks = add_indicators(test_stocks)

    # Добавление целевой переменной
    train_stocks = add_target(train_stocks)
    test_stocks = add_target(test_stocks)

    # Удаление строк с NaN (вызвано индикаторами)
    train_stocks = train_stocks.dropna()
    test_stocks = test_stocks.dropna()

    # Сохранение обработанных данных
    train_stocks.to_csv(train_stocks_with_features_path, index=False)
    test_stocks.to_csv(test_stocks_with_features_path, index=False)
    print("Stock data processed and saved.")

# Выполнение кода
if __name__ == "__main__":
    features_engineering()
