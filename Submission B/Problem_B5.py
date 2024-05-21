# ============================================================================================
# PROBLEM B5
#
# Build and train a neural network model using the Daily Max Temperature.csv dataset.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://github.com/jbrownlee/Datasets
#
# Desired MAE < 0.2 on the normalized dataset.
# ============================================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import urllib


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_B5():
    data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv'
    urllib.request.urlretrieve(data_url, 'daily-max-temperatures.csv')

    time_step = []
    temps = []

    with open('daily-max-temperatures.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            temps.append(  row[1]) # YOUR CODE HERE
            time_step.append(  row[0]) # YOUR CODE HERE
            step=step + 1

    series=np.array(temps, dtype=float)# YOUR CODE HERE

    # Normalization Function. DO NOT CHANGE THIS CODE
    min=np.min(series)
    max=np.max(series)
    series -= min
    series /= max
    time=np.array(time_step)

    # DO NOT CHANGE THIS CODE
    split_time=2500

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size=64
    batch_size=256
    shuffle_buffer_size=1000

    train_set=windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    print(train_set)
    print(x_train.shape)

    val_set = windowed_dataset(
        x_valid, window_size, batch_size, shuffle_buffer_size)

    model=tf.keras.models.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    # YOUR CODE HERE
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.fit(train_set, epochs=500, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], validation_data=val_set)

    # Plot the predictions against the actual values
    test_set = windowed_dataset(
        x_valid, window_size, batch_size, shuffle_buffer_size)
    test_pred = model.predict(test_set).flatten()

    # Take the first 500 samples from the test set
    test_pred = test_pred[:500]
    test_truth = x_valid[:500]
    plt.plot(test_truth, label='actual')
    plt.plot(test_pred, label='predicted')
    plt.legend()
    plt.show()
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B5()
    model.save("model_B5.h5")