# Intelligent Web Application for Investment Portfolio Analysis

![Technologies](https://img.shields.io/badge/Technologies-Python%20|%20FastAPI%20|%20React%20|%20TensorFlow%20|%20CatBoost-informational)

## Project Overview

This web application allows individual investors to analyze and manage their investment portfolios using:
- classical statistical methods (e.g., VaR, Expected Shortfall),
- artificial intelligence models (LSTM, Bayesian LSTM, CNN+Transformer, CatBoost).

Key features include:
- Portfolio and asset risk analysis (classical and AI-based).
- Asset price forecasting.
- Asset recommendation system.
- Portfolio optimization (Markowitz and AI-enhanced).
- Interactive charts and reports.

## Technologies

- **Backend**: Python, FastAPI, TensorFlow, CatBoost, scikit-learn
- **Frontend**: React, ApexCharts
- **Database**: PostgreSQL (SQLAlchemy)
- **Docker**: for backend and database deployment
- **Machine Learning**: LSTM, Bayesian LSTM, CNN+Transformer, CatBoost

## Getting Started

## Backend (FastAPI)

1. Clone the repository:
   ```bash
   git clone https://github.com/PatrykCzudak/projekt_dyplomowy.git
   cd projekt_dyplomowy
2. Run the backend using Docker:
   '''bash
   docker compose up --build
3. Frontend (React):
   '''bash
   cd frontend
   npm install
   npm run dev

## Historical Data
Historical data is fetched from Yahoo Finance using scripts in the AI/ directory and backend/app/services/yahoo.py.

## AI Models
  - LSTM: classifies asset risk using rolling windows of market data.
  - Bayesian LSTM: predicts expected returns and uncertainty.
  - CatBoost: recommends assets based on technical indicators.
  - CNN+Transformer: forecasts returns using sequential features.

## Application Features
  - Portfolio Management: manage portfolio, add transactions, view charts.
  - Transaction History: view all transactions with filtering options.
  - Charts: visualize historical prices and AI forecasts.
  - Prices: view current and forecasted prices (Bayesian LSTM).
  - Risk: analyze asset and portfolio risk using classical and AI methods.
  - Portfolio Optimization: optimize the portfolio using Markowitz and AI-based methods.
  - 
## Future Improvements
  - Incorporate macroeconomic data into AI models.
  - Implement automatic hyperparameter tuning.
  - Integrate additional fundamental indicators.