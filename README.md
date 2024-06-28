# Bitcoin Price Prediction using Time Series

This project aims to predict the future value of Bitcoin using time series analysis. Leveraging various machine learning and deep learning techniques, we strive to forecast Bitcoin prices with high accuracy.

## Overview

The goal of this project is to build a model that can predict Bitcoin prices based on historical data. Time series forecasting is employed to analyze patterns and trends in the data, allowing for predictions of future values.

## Features

- Data download and preprocessing: Automated data retrieval, handling missing values, normalization, and feature engineering.
- Model selection: Implementation of various time series forecasting models including ARIMA, LSTM, and Prophet.
- Model evaluation: Assessing model performance using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
- Visualization: Plotting historical and predicted prices for comparison.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lucascarmu/BTC-Price-Predict.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BTC-Price-Predict
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset:
   - Run the data download script to retrieve the latest Bitcoin price data:
     ```bash
     python download_data.py
     ```

2. Preprocess the data:
   - Ensure the downloaded data is saved in the `data/` directory.
   - Run the preprocessing script:
     ```bash
     python preprocess.py
     ```

3. Train the model:
   ```bash
   python train.py
   ```

4. Make predictions:
   ```bash
   python predict.py
   ```

5. Visualize the results:
   ```bash
   python visualize.py
   ```

## Project Structure

- `data/`: Directory for the dataset.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `scripts/`: Python scripts for downloading data, preprocessing, training, prediction, and visualization.
- `models/`: Directory to save trained models.
- `requirements.txt`: List of required packages.

---

For any questions or inquiries, please contact lucascarmusciano@gmail.com.
