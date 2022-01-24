import pandas as pd
import numpy as np
import nltk
import re
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess, loop_pw
from dataset import Dataset
from TNet_LF import tnet_lf
from tensorflow_addons.metrics import F1Score

path_to_data = 'data/train.csv'
split = 0.9
embedding_dim = 300
path_to_glove = 'data/embedding/glove.6B.300d.txt'
epoch=20
batch_size=64


dataset = Dataset(split, path_to_data)


tokenizer = dataset.create_tokenizer()

num_tokens = len(tokenizer.word_index) + 1

_, train_sequences, val_sequences = dataset.create_sequences(tokenizer)
_, train_aspect, val_aspect = dataset.create_sequences_aspect(tokenizer)
_, train_label, val_label = dataset.create_labels()

pw = dataset.create_pw()
train_pw = pw[:int(split * len(pw))]
val_pw = pw[int(split * len(pw)):]

# Create dataset

train_inputs = tf.data.Dataset.from_tensor_slices((train_sequences, train_aspect, train_pw))
val_inputs = tf.data.Dataset.from_tensor_slices((val_sequences, val_aspect, val_pw))


train_targets = tf.data.Dataset.from_tensor_slices(train_label)
val_target = tf.data.Dataset.from_tensor_slices(val_label)

train_data = tf.data.Dataset.zip((train_inputs, train_targets))
val_data = tf.data.Dataset.zip((val_inputs, val_target))

train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)



# Get the embedding matrix
embedding_matrix = dataset.create_embedding_matrix(tokenizer, path_to_glove, embedding_dim)

# Create model
model = tnet_lf(num_tokens, embedding_matrix)

# Compile model
model.compile(loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3),
             optimizer = tf.keras.optimizers.Adam(),
             metrics=['accuracy', F1Score(3, average='macro')])


# Fit the model
model.fit(train_data, epochs=epoch, validation_data=val_data)
