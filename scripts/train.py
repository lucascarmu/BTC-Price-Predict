import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras import layers
import config

# Load the datasets
train_dataset = tf.data.Dataset.load('./data/train_dataset')
test_dataset = tf.data.Dataset.load('./data/test_dataset')


# Create an ensemble of models
ensemble_models = []
num_iter = 3
num_epochs = 500
loss_fns = ["mae", "mse", "mape"]

# Directory to save the models
save_dir = './models/ensemble/'

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create num_iter number of models per loss function
for i in range(num_iter):
    for loss_function in loss_fns:
        print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}")

        # Construct a simple model
        model = tf.keras.Sequential([
            layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
            layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
            layers.Dense(config.HORIZON)
        ])

        # Compile simple model with current loss function
        model.compile(loss=loss_function,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["mae", "mse"])

        # Repeat the dataset for sufficient steps per epoch
        train_dataset_repeated = train_dataset.repeat()
        test_dataset_repeated = test_dataset.repeat()
        
        # Fit model
        model.fit(train_dataset_repeated,
                  epochs=num_epochs,
                  steps_per_epoch=1,
                  validation_data=test_dataset_repeated,
                  validation_steps=1,
                  verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                              patience=200,
                                                              restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                  patience=100,
                                                                  verbose=1)])

        # Save the fitted model
        model_save_path = os.path.join(save_dir, f'ensemble_model_{i}_{loss_function}.keras')
        model.save(model_save_path)
        
        # Append fitted model to list of ensemble models
        ensemble_models.append(model)