from fastapi import FastAPI
from app.config import settings
import tensorflow as tf
import os

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

save_dir = './models/ensemble/'
loaded_ensemble_models = [
    tf.keras.models.load_model(os.path.join(save_dir, file_name)) 
    for file_name in os.listdir(save_dir) if file_name.endswith('.keras')
]

test_dataset = tf.data.Dataset.load('./data/test_dataset')


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the BTC Prediction API"}

app.loaded_ensemble_models = loaded_ensemble_models
app.test_dataset = test_dataset

from app.api.endpoints import router as api_router
app.include_router(api_router, prefix="/api")