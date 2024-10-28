import sys
import os
import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.config import settings

# Load the datasets
train_dataset = tf.data.Dataset.load('./data/train_dataset')
test_dataset = tf.data.Dataset.load('./data/test_dataset')

# Create an ensemble of models
ensemble_models = []
num_iter = 6
num_epochs = 100
loss_fns = ["mae", "mse", "mape"]

# Directory to save the models
save_dir = './models/ensemble/'

# Calculate total number of elements
num_elements_train = sum(1 for _ in train_dataset.unbatch())

# Calculate steps per epoch
steps_per_epoch = num_elements_train // settings.BATCH_SIZE

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to create a new model
def create_model():
    model = tf.keras.Sequential([
        layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
        layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
        layers.Dense(settings.HORIZON)
    ])
    return model

# Create or load models for each loss function
for i in range(num_iter):
    for loss_function in loss_fns:
        model_save_path = os.path.join(save_dir, f'ensemble_model_{i}_{loss_function}.keras')
        
        print(f"Creating new model: ensemble_model_{i}_{loss_function}")
        model = create_model()
        model.compile(loss=loss_function,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae", "mse"])
        
        model.fit(train_dataset.repeat(),
                  epochs=num_epochs,
                  steps_per_epoch=steps_per_epoch,
                  verbose=1,
                  validation_data=test_dataset,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                              patience=50,
                                                              restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                  patience=25,
                                                                  verbose=1)])
        
        # Save the fitted model
        model.save(model_save_path)
        
        # Append fitted model to list of ensemble models
        ensemble_models.append(model)