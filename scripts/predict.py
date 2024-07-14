import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf # type: ignore
from scipy.stats import norm # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import config
from datetime import datetime
from utils import make_ensemble_preds

HORIZON = config.HORIZON
WINDOW_SIZE = config.WINDOW_SIZE


def predict(days, estimation):
    
    test_dataset = tf.data.Dataset.load('./data/test_dataset')
    save_dir = './models/ensemble/' # Directory where the models are saved

    # List to hold loaded models
    loaded_ensemble_models = []
    # List of predicts
    preds = []

    # Load each model
    for file_name in os.listdir(save_dir):
        if file_name.endswith('.keras'):
            model_path = os.path.join(save_dir, file_name)
            loaded_model = tf.keras.models.load_model(model_path)
            loaded_ensemble_models.append(loaded_model)
    
    # Get the last {HORIZON} values
    for X_batch, y_batch in iter(test_dataset):
        last_X_test, last_y_test = X_batch, y_batch
    
    last_X_test = last_X_test[-1,:HORIZON-2]
    last_y_test = last_y_test[-1]
    
    last_days = np.append(last_X_test, last_y_test)
    
    print(f"Calculating predictions for the next {days} days with estimation value: {estimation}...")
    for day in range(days):
        ensemble_preds = make_ensemble_preds(ensemble_models=loaded_ensemble_models,
                                            data=np.array(last_days).reshape(1, 7, 1))
        ensemble_median = np.median(ensemble_preds, axis=0)
        preds.append(ensemble_preds)
        
        print(f"{last_days} -> {ensemble_median}")
        last_days = last_days[1:]
        last_days = np.append(last_days, ensemble_median)
    
    dates = pd.date_range(start=datetime.today(), periods=days, freq='D').normalize()    

    # Find upper and lower bounds of ensemble predictions
    def get_upper_lower(preds, confidence_level=0.95):
        # Calculate the z-score for the given confidence level
        z_score = norm.ppf((1 + confidence_level) / 2)
    
        print(f"Z-SCORE: {z_score}")
        std = tf.math.reduce_std(preds, axis=1)
        interval = z_score * std

        preds_mean = tf.reduce_mean(preds, axis=1)
        lower, upper = preds_mean - interval, preds_mean + interval
        return lower, upper
    
    lower, upper = get_upper_lower(preds=preds,
                                   confidence_level=confidence_level)
    
    # Create a DataFrame with dates, median Bitcoin values, and confidence intervals
    df = pd.DataFrame({
        'Date': dates,
        'Bitcoin_Value': np.median(preds, axis=1),
        'Lower_CI': lower.numpy(),
        'Upper_CI': upper.numpy()
    })
    # Set the date column as the index of the DataFrame
    df.set_index('Date', inplace=True)
    
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict script with days and estimation value.')
    parser.add_argument('-d', '--days', type=int, required=True, help='Number of days for prediction')
    parser.add_argument('-c', '--confidence', type=float, required=True, help='Estimation value (between 0 and 1)')
    
    args = parser.parse_args()
    
    days = args.days
    confidence_level = args.confidence
    
    if not (0 <= confidence_level <= 1):
        print("Estimation value must be between 0 and 1.")
        exit(1)
    
    predict(days, confidence_level)