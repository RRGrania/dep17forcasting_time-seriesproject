# Department 17 Sales Forecasting Project 📈

This project performs **time series forecasting** and comparative evaluation using traditional statistical methods, machine learning, deep learning, and modern forecasting tools like Prophet. It specifically focuses on **Department 17** from a larger retail dataset.

---

## 📁 Dataset
- Original dataset: `project_dataset.csv`
- Extracted subset: `department_17_sales.csv`

---

## Modeling Techniques

### 1. Time Series Analysis
- **Visualization**
- **Trend & Seasonality Analysis**
- **ACF/PACF**
- **Decomposition**

###  2. Exponential Smoothing Models
- Holt-Winters
- Holt Linear Trend

###  3. ARIMA Family
- ARIMA
- SARIMA
- SARIMAX

###  4. Machine Learning Regressors
- Random Forest
- XGBoost  
(with feature engineering for temperature, fuel price, holidays, etc.)

###  5. Deep Learning Models
- LSTM
- GRU
- CNN

###  6. Facebook Prophet
- Including external regressors: temperature, fuel price, holidays

### 💡 7. Innovative Technique
- **DTW Similarity-based forecasting**
- **Ensemble Stacking (meta-learner using Ridge Regression)**

---

## ✅ Model Comparison Results

| Model              |   RMSE   |    MAE   |   MAPE   |
|--------------------|----------|----------|----------|
| **Ensemble Stacking** | 443.11 | 357.26 | 4.48% |
| Random Forest       | 463.25 | 376.53 | 4.65% |
| XGBoost             | 649.10 | 486.90 | 5.83% |
| Holt Linear         | 814.53 | 673.98 | 8.49% |
| GRU                 | 914.73 | 765.04 | 10.00% |
| LSTM                | 926.53 | 734.63 | 9.51% |
| SARIMA              | 946.78 | 791.35 | 9.90% |
| Holt-Winters        | 952.35 | 714.43 | 9.28% |
| CNN                 | 986.62 | 756.17 | 9.74% |
| Prophet             | 1459.67 | 1308.56 | 16.12% |
| SARIMAX             | 1521.61 | 1183.58 | 15.12% |
| ARIMA               | 1568.10 | 1384.36 | 18.37% |
| DTW Similarity      | 3739.66 | 2817.62 | 36.65% |

> ✅ **Best Model:** Ensemble Stacking with the lowest RMSE, MAE, and MAPE

---

## 💾 Model Saving
- Final model saved using `joblib`:
```python
joblib.dump(meta_learner, 'ensemble_stacking_model.pkl')
