import os
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
from app.utils import evaluate_preds, make_ensemble_preds

# Create the output directory if it doesn't exist
output_dir = './outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

test_dataset = tf.data.Dataset.load('./data/test_dataset')
save_dir = './models/ensemble/' # Directory where the models are saved

# List to hold loaded models
loaded_ensemble_models = []

# Load each model
for file_name in os.listdir(save_dir):
    if file_name.endswith('.keras'):
        model_path = os.path.join(save_dir, file_name)
        loaded_model = tf.keras.models.load_model(model_path)
        loaded_ensemble_models.append(loaded_model)


# Create a list of ensemble predictions
ensemble_preds = make_ensemble_preds(ensemble_models=loaded_ensemble_models,
                                     data=test_dataset)

ensemble_median = np.median(ensemble_preds, axis=0)

# Separate features and labels from the test dataset
X_test = []
y_test = []
for X_batch, y_batch in test_dataset.as_numpy_iterator():
    X_test.append(X_batch)
    y_test.append(y_batch)
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

ensemble_results = evaluate_preds(y_true=y_test,
                                  y_pred=ensemble_median)

# Find upper and lower bounds of ensemble predictions
def get_upper_lower(preds):
    std = tf.math.reduce_std(preds, axis=0)
    interval = 1.96 * std # https://en.wikipedia.org/wiki/1.96 

    preds_mean = tf.reduce_mean(preds, axis=0)
    lower, upper = preds_mean - interval, preds_mean + interval
    return lower, upper

# Get the upper and lower bounds of the 95% 
lower, upper = get_upper_lower(preds=ensemble_preds)

# Convert X_test to a pandas DataFrame
X_test_df = pd.DataFrame(X_test)
offset = 450
dates = pd.date_range(end=datetime.today(), periods=len(X_test), freq='D')
X_test_df['Date'] = dates
X_test_df.set_index('Date', inplace=True)
X_test_df.sort_index(ascending=True, inplace=True)

# Plot the median of our ensemble preds along with the prediction intervals (where the predictions fall between)
plt.figure(figsize=(10, 7))
plt.plot(X_test_df.index[offset:], y_test[offset:], "g", label="Test Data")
plt.plot(X_test_df.index[offset:], ensemble_median[offset:], "k-", label="Ensemble Median")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.fill_between(X_test_df.index[offset:], 
                 (lower)[offset:], 
                 (upper)[offset:], label="Prediction Intervals")
plt.legend(loc="upper left", fontsize=14)


# Save the plot to the output directory
plot_path = os.path.join(output_dir, 'ensemble_predictions.png')
plt.savefig(plot_path)
plt.show()
