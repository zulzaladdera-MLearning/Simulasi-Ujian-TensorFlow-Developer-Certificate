# =====================================================================================================
# PROBLEM C4
# ZULZA LADDERA ARIPIN
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.76 and logs.get('val_accuracy')>0.76):
            print("\nTarget telah dicapai, berhenti training !!!")
            self.model.stop_training = True



def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json', 'r') as json_read:
        getdata = json.load(json_read)

    for i in getdata:
        sentences.append(i['headline'])
        labels.append(i['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = np.array(labels[:training_size])
    test_sentences = sentences[training_size:]
    test_labels = np.array(labels[training_size:])

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size,
                           oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences) # YOUR CODE HERE

    training_seq = tokenizer.texts_to_sequences(train_sentences)
    train_sentences_pad = pad_sequences(training_seq,
                                        maxlen=max_length,
                                        truncating=trunc_type,
                                        padding=padding_type)

    testing_seq = tokenizer.texts_to_sequences(test_sentences)
    test_sentences_pad = pad_sequences(testing_seq,
                                       maxlen=max_length,
                                       truncating=trunc_type,
                                       padding=padding_type)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_sentences_pad,
              train_labels,
              epochs=20,
              validation_data=(test_sentences_pad, test_labels),
              callbacks=[myCallback()])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
