from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np

# Пути
train_data_path = "../data/train_stocks_with_features.csv"
cv_metric_path = "../results/cross-validation/ml_metrics_train.csv"
model_path = "../results/selected-model/selected_model.pkl"
params_path = "../results/selected-model/selected_model.txt"
blocking_plot_path = "../results/cross-validation/blocking_plot.png"
time_series_plot_path = "../results/cross-validation/time_series_plot.png"

# Создание директорий
os.makedirs("../results/cross-validation", exist_ok=True)
os.makedirs("../results/selected-model", exist_ok=True)

# Загрузка данных
try:
    train_data = pd.read_csv(train_data_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Train data file not found: {e}")

# Проверка данных
if train_data.isnull().sum().sum() > 0:
    print("Warning: Missing values detected in train data.")

# Выбор признаков и целевой переменной
X_train = train_data[["bb_high", "bb_low", "rsi", "macd", "macd_signal"]]
y_train = train_data["target"]

# Функция для построения графиков фолдов
def plot_folds(cv, X, y, save_path, title):
    plt.figure(figsize=(10, 6))
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        plt.plot(train_idx, [fold_idx + 1] * len(train_idx), '.', label=f"Train Fold {fold_idx + 1}" if fold_idx == 0 else "")
        plt.plot(val_idx, [fold_idx + 1] * len(val_idx), 'x', label=f"Validation Fold {fold_idx + 1}" if fold_idx == 0 else "")
    plt.xlabel("Data Index")
    plt.ylabel("Fold Number")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# Blocking cross-validation
blocking_cv = KFold(n_splits=10, shuffle=False)
plot_folds(blocking_cv, X_train, y_train, blocking_plot_path, "Blocking Cross-validation Folds")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=10)
plot_folds(tscv, X_train, y_train, time_series_plot_path, "Time Series Cross-validation Folds")

# Модель и сетка гиперпараметров
model = GradientBoostingClassifier()
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8],
    "min_samples_split": [2, 10]
}

# GridSearchCV
grid_search = GridSearchCV(
    model,
    param_grid,
    scoring={"AUC": "roc_auc", "Accuracy": "accuracy"},
    refit="Accuracy",
    cv=blocking_cv,
    verbose=1,
    return_train_score=True,
    n_jobs=-1
)

# Обучение модели
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

# Сохранение лучшей модели
joblib.dump(grid_search.best_estimator_, model_path)
with open(params_path, "w") as f:
    f.write(str(grid_search.best_params_))

# Сохранение метрик
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(cv_metric_path, index=False)


print("Grid search and cross-validation completed successfully.")
