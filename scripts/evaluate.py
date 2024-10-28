import os
import json
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

evaluation_results = evaluate_preds(y_true=y_test,
                                  y_pred=ensemble_median)

# Save the results to a json file
with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
    json.dump(evaluation_results, f)
    
print("Ensemble model evaluation results saved to 'outputs/evaluation_results.json'")