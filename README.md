# Fina# 📈 Finance Forecast Project

This project analyzes and forecasts Apple (AAPL) stock price trends using Python and visualizes insights with Tableau.

---

## 📊 Project Overview

- Downloaded historical Apple stock data (2020–2024) via `yfinance`
- Applied moving average strategies (SMA20, SMA50, SMA200)
- Implemented and evaluated trading strategy (Golden/Death Cross)
- Built machine learning models (Linear Regression, Random Forest, XGBoost) for price prediction
- Exported cleaned dataset for Tableau dashboard visualization

---

## 🧠 ML Models Compared

| Model              | MSE     | R² Score |
|-------------------|---------|----------|
| Linear Regression | ~97     | ~0.99    |
| Random Forest     | ~721    | ~0.79    |
| XGBoost           | ~767    | ~0.14    |

---

## 🛠️ Tools & Technologies

- Python (Pandas, Scikit-learn, XGBoost, yfinance, matplotlib)
- Tableau Public (for dashboard visualization)
- Git & GitHub

---

## 📂 Key Files

| File Name              | Description                                      |
|------------------------|--------------------------------------------------|
| `main.py`              | Full end-to-end stock analysis and prediction script |
| `AAPL_dashboard_data.csv` | Cleaned dataset for Tableau visualization       |
| `AAPL_stock.csv`       | Raw downloaded data                              |
| `requirements.txt`     | Dependencies list                                |

---

## 📊 Tableau Dashboard

> Published on Tableau Public: [Insert Your Tableau Link Here]

---

## 📌 Future Improvements

- Add hyperparameter tuning for better model accuracy  
- Expand to other stock symbols or sectors  
- Integrate real-time data for dynamic updates  
