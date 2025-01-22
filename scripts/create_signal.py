import pandas as pd
import joblib
import os

# Paths
model_path = "../results/selected-model/selected_model.pkl"
test_data_path = "../data/test_stocks_with_features.csv"
signal_output_path = "../results/selected-model/ml_signal.csv"

# Check for files
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data not found at {test_data_path}")

# Loading the model
model = joblib.load(model_path)

# Loading test data
test_data = pd.read_csv(test_data_path)
X_test = test_data[["bb_high", "bb_low", "rsi", "macd", "macd_signal"]]  # Selection of features

# Model Predictions
predictions = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Forming a DataFrame with signals
ml_signal = test_data[["date", "name"]].copy()  # Preserving metainformation
ml_signal["signal"] = predictions  # Adding model signals

# Saving signal to file
ml_signal.to_csv(signal_output_path, index=False)
print(f"Machine learning signal saved to {signal_output_path}")
