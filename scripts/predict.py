import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
        preds.append(ensemble_median)
        
        print(f"{last_days} -> {ensemble_median}")
        last_days = last_days[1:]
        last_days = np.append(last_days, ensemble_median)
    
    dates = pd.date_range(start=datetime.today(), periods=days, freq='D').normalize()    
    # Crear un DataFrame con las fechas y los valores de Bitcoin
    df = pd.DataFrame({
        'Date': dates,
        'Bitcoin_Value': preds
    })

    # Establecer la columna de fechas como el índice del DataFrame
    df.set_index('Date', inplace=True)
    
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict script with days and estimation value.')
    parser.add_argument('-d', '--days', type=int, required=True, help='Number of days for prediction')
    parser.add_argument('-e', '--estimation', type=float, required=True, help='Estimation value (between 0 and 1)')
    
    args = parser.parse_args()
    
    days = args.days
    estimation = args.estimation
    
    if not (0 < estimation <= 1):
        print("Estimation value must be between 0 and 1.")
        exit(1)
    
    predict(days, estimation)