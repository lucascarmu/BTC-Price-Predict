# scripts/utils.py
# type: ignore
import requests as request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import tensorflow as tf


def get_binance_klines(symbol, interval, start_str, end_str=None, max_retries=3):
    url = "https://api.binance.com/api/v3/klines"
    start_time = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_time = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else None
    all_data = []
    
    retries = 0
    while start_time < end_time and retries < max_retries:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        try:
            response = request.get(url, params=params)
            if response.status_code == 451:
                print(f"Error: {response.status_code}, {response.json()}")
                break
            
            data = response.json()
            if response.status_code != 200 or not data:
                print(f"Error: {response.status_code}, {response.text}")
                retries += 1
                time.sleep(2 ** retries)  # Exponential backoff
                continue
            
            all_data.extend(data)
            start_time = data[-1][0] + 1  # Move to the next time window
            retries = 0  # Reset retries if the request was successful
        except request.exceptions.RequestException as e:
            print(f"Request error: {e}")
            retries += 1
            time.sleep(2 ** retries)  # Exponential backoff
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns={'timestamp': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    return df

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)
  
def custom_plot_series(y_train, y_test, y_pred, start=0, labels=["y_train", "y_test", "y_pred"], marker_size=5):
    """
    Plots time series data for training, testing, and predicted values.

    Parameters:
    - y_train: pandas Series
        The training data series.
    - y_test: pandas Series
        The testing data series.
    - y_pred: pandas Series
        The predicted data series.
    - start: int, optional (default=0)
        The starting index from which to plot the series.
    - labels: list of str, optional (default=["y_train", "y_test", "y_pred"])
        The labels for the training, testing, and predicted series.
    - marker_size: int, optional (default=5)
        The size of the markers in the plot.

    Returns:
    None. Displays the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the series
    plt.plot(y_train[start:].index, y_train[start:], marker='o', markersize=marker_size, label=labels[0])
    plt.plot(y_test[start:].index, y_test[start:], marker='o', markersize=marker_size, label=labels[1])
    plt.plot(y_pred[start:].index, y_pred[start:], marker='o', markersize=marker_size, label=labels[2])
    
    plt.legend()

    plt.title("Time Series Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    plt.show()
    
    
def evaluate_preds(y_true, y_pred):
    """
    Evaluate the predictions using common regression metrics.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    dict: A dictionary with the calculated metrics.
    """
    
    # Convert to numpy arrays if not already
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Calculate the metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Print the metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    # Return the metrics in a dictionary
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
    
    return metrics


def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.
    
    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    # print(f"Window step:\n {window_step}")
    
    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
    # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")
    
    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]
    
    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels

def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels

def make_ensemble_preds(ensemble_models, data):
        ensemble_preds = []
        for model in ensemble_models:
            preds = model.predict(data)
            ensemble_preds.append(preds)
        return tf.constant(tf.squeeze(ensemble_preds))
    