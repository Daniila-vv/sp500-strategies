import pandas as pd
import os
import matplotlib.pyplot as plt

# Пути
signal_path = "../results/selected-model/ml_signal.csv"
historical_data_path = "../data/test_historical.csv"
strategy_results_path = "../results/strategy/strategy_results.csv"
cumulative_plot_path = "../results/strategy/cumulative_returns.png"

# Проверка наличия файлов
if not os.path.exists(signal_path):
    raise FileNotFoundError(f"Signal file not found at {signal_path}")
if not os.path.exists(historical_data_path):
    raise FileNotFoundError(f"Historical data file not found at {historical_data_path}")

# Загрузка данных
ml_signal = pd.read_csv(signal_path)
historical_data = pd.read_csv(historical_data_path)

# Преобразование даты в формат datetime
ml_signal['date'] = pd.to_datetime(ml_signal['date'])
historical_data['date'] = pd.to_datetime(historical_data['date'])

# Объединение данных
merged_data = pd.merge(
    ml_signal,
    historical_data,
    on="date",
    how="inner"
)

if merged_data.empty:
    raise ValueError("Merged data is empty. Please check signal and historical data alignment.")

# Реализация стратегии
merged_data['position'] = merged_data['signal'].apply(lambda x: 1 if x > 0.5 else -1)
merged_data['daily_return'] = merged_data.groupby('name')['close'].pct_change()
merged_data['strategy_return'] = merged_data['position'].shift(1) * merged_data['daily_return']

# Группируем по тикеру
cumulative_strategy_return = merged_data.groupby('name')['strategy_return'].apply(lambda x: (1 + x.fillna(0)).cumprod())
cumulative_market_return = merged_data.groupby('name')['daily_return'].apply(lambda x: (1 + x.fillna(0)).cumprod())

# Приведение индексов к исходному DataFrame
merged_data['cumulative_strategy_return'] = cumulative_strategy_return.reset_index(level=0, drop=True)
merged_data['cumulative_market_return'] = cumulative_market_return.reset_index(level=0, drop=True)

# Сохранение результатов стратегии
merged_data.to_csv(strategy_results_path, index=False)
print(f"Strategy results saved to {strategy_results_path}")

# Построение графика кумулятивной доходности
plt.figure(figsize=(12, 6))
plt.plot(
    merged_data.groupby('date')['cumulative_strategy_return'].mean(),
    label='Strategy Return (Mean)'
)
plt.plot(
    merged_data.groupby('date')['cumulative_market_return'].mean(),
    label='Market Return (Mean)'
)
plt.legend()
plt.title('Cumulative Returns: Strategy vs. Market')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid()
plt.savefig(cumulative_plot_path)
print(f"Cumulative returns plot saved to {cumulative_plot_path}")
plt.close()
