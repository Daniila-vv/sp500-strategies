
# Report: SP500 Trading Strategy

---

#### **Project Goal**
To develop a trading strategy based on SP500 data, analyze the performance of the strategy compared to the market returns, and meet the requirements outlined in the project task.

---

### **Assessment of Project Requirements**

| **Requirement**                                       | **Met?**       | **Details of Implementation**                                                                                                 |
|-------------------------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| **1. Implement a strategy based on Gradient Boosting.** | ✅ Yes         | A Gradient Boosting model from `Scikit-learn` was used for predictions.                                                       |
| **2. Optimize model hyperparameters.**               | ✅ Yes         | Hyperparameters were optimized using cross-validation:<br> `{'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.8}`. |
| **3. Perform 10-fold cross-validation.**             | ✅ Yes         | Cross-validation with 10 folds was implemented. Metrics for each fold were computed, and the mean AUC was calculated for train and test sets. |
| **4. Plot AUC metrics for train and validation sets.** | ✅ Yes         | AUC metrics for train and test sets were plotted. The graph demonstrates no significant overfitting (screenshot attached).    |
| **5. Use RSI, MACD, and Bollinger Bands.**           | ✅ Yes         | All three indicators were used as features in the model. Top 5 most important features include RSI, MACD Signal, Bollinger Bands (Low and High). |
| **6. Plot cumulative returns.**                      | ✅ Yes         | The "Cumulative Returns: Strategy vs. Market" graph was generated.                                                           |
| **7. Compare the strategy to the market.**           | ✅ Yes         | The graph shows that the strategy underperforms the market returns but still achieves positive returns.                       |

---

### **Project Execution Steps**

#### 1. **Data Preparation**
- SP500 data was loaded and preprocessed.
- Initial data analysis and cleaning were performed.
- Features were engineered, including RSI, MACD, and Bollinger Bands (Low and High).
- A target variable was created based on future price changes.

#### 2. **Model Training and Optimization**
- Gradient Boosting from `Scikit-learn` was selected as the model.
- Hyperparameters were optimized using 10-fold cross-validation.
- Best parameters found:
  - **learning_rate**: 0.1
  - **max_depth**: 3
  - **min_samples_split**: 2
  - **n_estimators**: 100
  - **subsample**: 0.8
- Average AUC metrics:
  - **Train**: 0.527
  - **Test**: 0.514

#### 3. **Feature Importance Analysis**
- A feature importance graph was generated.
- Top 5 most important features:
  1. **RSI**: 37.34%
  2. **MACD Signal**: 18.90%
  3. **Bollinger Low**: 16.79%
  4. **Bollinger High**: 16.14%
  5. **MACD**: 10.83%

#### 4. **AUC Metrics Visualization**
- A graph comparing AUC metrics for train and test sets was generated. 
- The graph indicates stable test set performance, though lower than train metrics, highlighting areas for improvement.

#### 5. **Strategy Analysis**
- The "Cumulative Returns" graph was created:
  - The strategy achieves positive returns but underperforms the market.
  - The model shows potential for improvement through fine-tuning.

---

### **Project Outcomes**

| **Task**                                              | **Status**                                                                                                                   |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Implement Gradient Boosting model                    | ✅ Completed                                                                                                                |
| Perform cross-validation and compute AUC metrics     | ✅ Completed                                                                                                                |
| Visualize AUC metrics                                | ✅ Completed                                                                                                                |
| Analyze feature importance                           | ✅ Completed                                                                                                                |
| Compare cumulative returns to the market             | ✅ Completed                                                                                                                |
| Use RSI, MACD, and Bollinger Bands                   | ✅ Completed                                                                                                                |

---

### **Conclusions and Recommendations**

1. **Task Completion**: All project requirements were fulfilled.
2. **Potential for Improvement**:
   - Refine signal selection rules.
   - Introduce risk management mechanisms (e.g., stop-loss, take-profit).
   - Experiment with alternative models, such as XGBoost or Random Forest.
3. **Recommendations**:
   - Focus on improving strategy profitability.
   - Enhance feature engineering by incorporating additional indicators or features.

---

### **Summary**
The project was completed according to all requirements. The model and strategy have potential for further improvement to achieve superior performance.
