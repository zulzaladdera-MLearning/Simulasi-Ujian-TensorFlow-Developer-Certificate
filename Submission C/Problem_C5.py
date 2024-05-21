# ============================================================================================
# PROBLEM C5
# ZULZA LADDERA ARIPIN
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model
# should be trained to predict the next 24 observations of the 7 variables.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
#
# Desired MAE < 0.1 on the normalized dataset.
# ============================================================================================

import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf

# This function downloads and extracts the dataset to the directory that contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change the URL)
def download_and_extract_data():
    url = 'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(n_past + n_future, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(n_past + n_future))
    dataset =dataset.shuffle(1000)
    dataset = dataset.map(lambda window: (window[:-n_future], window[-n_future:, :1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def solution_C5():
    download_and_extract_data()
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)

    N_FEATURES = 7

    data = df.values
    split_time = int(len(data) * 0.5)
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    x_train = data[:split_time]
    x_valid = data[split_time:]

    BATCH_SIZE = 32
    N_PAST = 24
    N_FUTURE = 24
    SHIFT = 1

    train_set = windowed_dataset(x_train, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)
    valid_set = windowed_dataset(x_valid, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if(logs.get('mae')<0.09 and logs.get('val_mae')<0.09):
                print("\nMAE training dan validasi telah lebih kecil dari 0.9")
                self.model.stop_training = True

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=[N_PAST, N_FEATURES]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(N_FUTURE)
    ])

    model.compile(loss='mae', metrics=['mae'], optimizer='adam')
    model.fit(train_set, epochs=10, validation_data=valid_set, callbacks=[myCallback()], verbose=1)
    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C5()
    model.save("model_C5.h5")