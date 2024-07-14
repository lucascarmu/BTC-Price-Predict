import shutil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow as tf
import config

HORIZON = config.HORIZON
WINDOW_SIZE = config.WINDOW_SIZE


df = pd.read_csv('./data/binance_btcusdt_price_history.csv')
btc_prices = pd.DataFrame(df['close']).rename(columns={'close': 'Price'})
btc_prices['Price'] = btc_prices['Price'].astype(float)

# Add windowed columns
btc_prices_nbeats = btc_prices.copy()
for i in range(WINDOW_SIZE):
  btc_prices_nbeats[f"Price+{i+1}"] = btc_prices_nbeats["Price"].shift(periods=i+1)

# Make features and labels
X = btc_prices_nbeats.dropna().drop("Price", axis=1)
y = btc_prices_nbeats.dropna()["Price"]

# Make train and test sets
split_size = int(len(X) * 0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]

# 1. Turn train and test arrays into tensor Datasets
train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

# 2. Combine features & labels
train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

# 3. Batch and prefetch for optimal performance
train_dataset = train_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Define file paths
train_dataset_file = './data/train_dataset/'
test_dataset_file = './data/test_dataset/'

# Remove old files if they exist
if os.path.exists(train_dataset_file):
    shutil.rmtree(train_dataset_file)
if os.path.exists(test_dataset_file):
    shutil.rmtree(test_dataset_file)

# Save the datasets
tf.data.Dataset.save(train_dataset, train_dataset_file)
tf.data.Dataset.save(test_dataset, test_dataset_file)