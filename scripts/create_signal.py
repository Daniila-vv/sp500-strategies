import pandas as pd
import joblib
import os

# Пути
model_path = "../results/selected-model/selected_model.pkl"
test_data_path = "../data/test_stocks_with_features.csv"
signal_output_path = "../results/selected-model/ml_signal.csv"

# Проверка наличия файлов
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data not found at {test_data_path}")

# Загрузка модели
model = joblib.load(model_path)

# Загрузка тестовых данных
test_data = pd.read_csv(test_data_path)
X_test = test_data[["bb_high", "bb_low", "rsi", "macd", "macd_signal"]]  # Выбор признаков

# Предсказания модели
predictions = model.predict_proba(X_test)[:, 1]  # Вероятность положительного класса

# Формирование DataFrame с сигналами
ml_signal = test_data[["date", "name"]].copy()  # Сохраняем метаинформацию
ml_signal["signal"] = predictions  # Добавляем сигналы модели

# Сохранение сигнала в файл
ml_signal.to_csv(signal_output_path, index=False)
print(f"Machine learning signal saved to {signal_output_path}")
