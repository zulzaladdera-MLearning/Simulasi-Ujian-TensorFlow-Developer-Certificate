# =====================================================================================
# PROBLEM A2
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================


import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import BatchNormalization

def solution_A2():
    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    TRAINING_DIR = 'data/horse-or-human'
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=RMSprop(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=20,
        verbose=1
    )

    print("Training Accuracy:", history.history['accuracy'][-1])

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    validation_generator = validation_datagen.flow_from_directory(
        'data/validation-horse-or-human',
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'
    )

    validation_loss, validation_accuracy = model.evaluate(validation_generator)
    print("Validation Accuracy:", validation_accuracy)

    return model
# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_A2()
    model.save("model_A2.h5")