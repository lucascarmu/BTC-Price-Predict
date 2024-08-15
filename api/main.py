from pydantic import BaseModel, Field
from scipy.stats import norm
import config
import sys
import os
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime
from io import BytesIO  # For handling image data in memory
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from utils import evaluate_preds, make_ensemble_preds

app = FastAPI()

HORIZON = config.HORIZON
WINDOW_SIZE = config.WINDOW_SIZE

# Load models once to avoid loading them on every request
save_dir = './models/ensemble/'
loaded_ensemble_models = [tf.keras.models.load_model(os.path.join(save_dir, file_name)) 
                          for file_name in os.listdir(save_dir) if file_name.endswith('.keras')]

# Load the test dataset
test_dataset = tf.data.Dataset.load('./data/test_dataset')

# Pydantic model for request body
class PredictionRequest(BaseModel):
    days: int = Field(..., description="Number of days to predict", ge=1)
    confidence_level: float = Field(1.0, description="Confidence level for the prediction", ge=0, le=1)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "days": 7,
                "confidence_level": 0.95
            }
        }
    }

@app.post("/predict/")
def predict(request: PredictionRequest):
    days = request.days
    confidence_level = request.confidence_level

    # Get the last {HORIZON} values
    for X_batch, y_batch in iter(test_dataset):
        last_X_test, last_y_test = X_batch, y_batch
    
    last_X_test = last_X_test[-1, :HORIZON-2]
    last_y_test = last_y_test[-1]
    
    last_days = np.append(last_X_test, last_y_test)
    
    preds = []
    for day in range(days):
        ensemble_preds = make_ensemble_preds(ensemble_models=loaded_ensemble_models,
                                             data=np.array(last_days).reshape(1, 7, 1))
        ensemble_median = np.median(ensemble_preds, axis=0)
        preds.append(ensemble_preds)
        
        last_days = last_days[1:]
        last_days = np.append(last_days, ensemble_median)
    
    dates = pd.date_range(start=datetime.today(), periods=days, freq='D').normalize()

    def get_upper_lower(preds, confidence_level=0.95):
        z_score = norm.ppf((1 + confidence_level) / 2)
        std = tf.math.reduce_std(preds, axis=1)
        interval = z_score * std

        preds_mean = tf.reduce_mean(preds, axis=1)
        lower, upper = preds_mean - interval, preds_mean + interval
        return lower, upper

    lower, upper = get_upper_lower(preds=preds, confidence_level=confidence_level)
    
    df = pd.DataFrame({
        'Date': dates,
        'Bitcoin_Value': np.median(preds, axis=1),
        'Lower_CI': lower.numpy(),
        'Upper_CI': upper.numpy()
    })
    df.set_index('Date', inplace=True)

    return df.to_dict(orient="index")


@app.get("/evaluate")
def evaluate_model():
    """Evaluate the model and return the results and the plot as a response."""

    # Create a list of ensemble predictions
    ensemble_preds = make_ensemble_preds(ensemble_models=loaded_ensemble_models, data=test_dataset)
    ensemble_median = np.median(ensemble_preds, axis=0)

    # Separate features and labels from the test dataset
    X_test, y_test = [], []
    for X_batch, y_batch in test_dataset.as_numpy_iterator():
        X_test.append(X_batch)
        y_test.append(y_batch)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Evaluate the ensemble model predictions
    ensemble_results = evaluate_preds(y_true=y_test, y_pred=ensemble_median)

    # Return JSON response with evaluation results
    return {"evaluation_results": ensemble_results}