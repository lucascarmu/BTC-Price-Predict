from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from datetime import datetime
from scipy.stats import norm
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import os

from app.main import app
from app.utils import make_ensemble_preds, evaluate_preds
from app.config import settings

router = APIRouter()

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

@router.post("/predict/")
def predict(request: PredictionRequest):
    days = request.days
    confidence_level = request.confidence_level

    test_dataset = app.test_dataset
    loaded_ensemble_models = app.loaded_ensemble_models

    for X_batch, y_batch in iter(test_dataset):
        last_X_test, last_y_test = X_batch, y_batch

    last_X_test = last_X_test[-1, :settings.HORIZON-2]
    last_y_test = last_y_test[-1]
    last_days = np.append(last_X_test, last_y_test)

    preds = []
    for day in range(days):
        ensemble_preds = make_ensemble_preds(loaded_ensemble_models, np.array(last_days).reshape(1, 7, 1))
        ensemble_median = np.median(ensemble_preds, axis=0)
        preds.append(ensemble_preds)

        last_days = last_days[1:]
        last_days = np.append(last_days, ensemble_median)

    dates = pd.date_range(start=datetime.today(), periods=days, freq='D').normalize()

    def get_upper_lower(preds, confidence_level=0.95):
        z_score = norm.ppf((1 + confidence_level) / 2)
        std = np.std(preds, axis=1)
        interval = z_score * std
        preds_mean = np.mean(preds, axis=1)
        return preds_mean - interval, preds_mean + interval

    lower, upper = get_upper_lower(preds=preds, confidence_level=confidence_level)

    df = pd.DataFrame({
        'Date': dates,
        'Bitcoin_Value': np.median(preds, axis=1),
        'Lower_CI': lower,
        'Upper_CI': upper
    })
    df.set_index('Date', inplace=True)

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.plot(df.index, df['Bitcoin_Value'], 'k-', label='Ensemble Median')
    plt.fill_between(df.index, df['Lower_CI'], df['Upper_CI'], color='b', alpha=0.3, label='Prediction Intervals')
    plt.title('Predicción de Bitcoin')
    plt.xlabel('Fecha')
    plt.ylabel('Valor de BTC')
    plt.legend(loc="upper left", fontsize=14)

    plot_path = os.path.join(output_dir, 'ensemble_predictions.png')
    plt.savefig(plot_path)
    plt.close()

    return df.to_dict(orient="index")

@router.get("/evaluate")
def evaluate_model():
    """Evaluate the model and return the results."""
    
    test_dataset = app.test_dataset
    loaded_ensemble_models = app.loaded_ensemble_models

    ensemble_preds = make_ensemble_preds(loaded_ensemble_models, test_dataset)
    ensemble_median = np.median(ensemble_preds, axis=0)

    X_test, y_test = [], []
    for X_batch, y_batch in test_dataset.as_numpy_iterator():
        X_test.append(X_batch)
        y_test.append(y_batch)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    ensemble_results = evaluate_preds(y_true=y_test, y_pred=ensemble_median)

    return {"evaluation_results": ensemble_results}