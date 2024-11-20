import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Пути
ml_metrics_path = "../results/cross-validation/ml_metrics_train.csv"
metric_train_path = "../results/cross-validation/metric_train.csv"
metric_plot_path = "../results/cross-validation/metric_train.png"
model_path = "../results/selected-model/selected_model.pkl"
feature_importance_path = "../results/cross-validation/top_10_feature_importance.csv"

# Создание metric_train.csv
def create_metric_train():
    try:
        # Загрузка метрик
        results = pd.read_csv(ml_metrics_path)

        # Создание нового файла metric_train.csv
        results.to_csv(metric_train_path, index=False)
        print(f"Файл {metric_train_path} создан.")
    except FileNotFoundError:
        print(f"Не удалось найти {ml_metrics_path}. Проверьте путь и попробуйте снова.")

# Создание графика метрик metric_train.png
def plot_metric_train():
    try:
        # Загрузка данных
        results = pd.read_csv(ml_metrics_path)

        # Построение графика метрик
        plt.figure(figsize=(10, 6))

        # График AUC
        plt.plot(results["mean_train_AUC"], label="Train AUC", marker="o")
        plt.plot(results["mean_test_AUC"], label="Validation AUC", marker="x")

        # Оформление графика
        plt.xlabel("Hyperparameter Combinations")
        plt.ylabel("AUC")
        plt.title("AUC Scores for Train and Validation")
        plt.legend()
        plt.grid(True)

        # Сохранение графика
        plt.savefig(metric_plot_path)
        plt.close()
        print(f"График метрик сохранен в {metric_plot_path}.")
    except FileNotFoundError:
        print(f"Не удалось найти {ml_metrics_path}. Проверьте путь и попробуйте снова.")
    except KeyError as e:
        print(f"Ошибка: отсутствует ключ {e} в данных метрик.")

# Создание top_10_feature_importance.csv
def create_feature_importance():
    try:
        # Загрузка модели
        best_model = joblib.load(model_path)

        # Создание DataFrame с важностью признаков
        feature_importance = pd.DataFrame({
            "Feature": ["bb_high", "bb_low", "rsi", "macd", "macd_signal"],
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        # Сохранение в файл
        feature_importance.to_csv(feature_importance_path, index=False)
        print(f"Файл {feature_importance_path} создан.")
    except FileNotFoundError:
        print(f"Не удалось найти модель {model_path}. Проверьте путь и попробуйте снова.")
    except AttributeError:
        print("Ошибка: модель не содержит важности признаков.")

# Основной запуск
if __name__ == "__main__":
    create_metric_train()
    plot_metric_train()
    create_feature_importance()
