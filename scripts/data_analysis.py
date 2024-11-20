import pandas as pd

# Пути к файлам
historical_prices_path = "../data/HistoricalPrices.csv"
all_stocks_data_path = "../data/all_stocks_5yr.csv"

# Загрузка данных
historical_prices = pd.read_csv(historical_prices_path)
all_stocks_data = pd.read_csv(all_stocks_data_path)

# Анализ структуры данных
print("Historical Prices Data")
print(historical_prices.info())
print(historical_prices.head())

print("\nAll Stocks Data")
print(all_stocks_data.info())
print(all_stocks_data.head())

# Проверка на пропуски
print("\nMissing values in Historical Prices Data:")
print(historical_prices.isnull().sum())

print("\nMissing values in All Stocks Data:")
print(all_stocks_data.isnull().sum())

# Проверка на дубли
print("\nDuplicates in Historical Prices Data:", historical_prices.duplicated().sum())
print("Duplicates in All Stocks Data:", all_stocks_data.duplicated().sum())
