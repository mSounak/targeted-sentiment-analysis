import pandas as pd
import numpy as np
import nltk
import re
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess, loop_pw
from sklearn.preprocessing import OneHotEncoder


class Dataset:

    def __init__(self, split, path_to_data):
        self.split = split
        self.path = path_to_data

    def create_dataset(self):
        df = pd.read_csv(self.path)

        df['text'] = df['text'].apply(preprocess)
        df['aspect'] = df['aspect'].apply(preprocess)

        train_df = df[: int(self.split * len(df))]
        val_df = df[int(self.split * len(df)): ]

        return df, train_df, val_df


    def create_tokenizer(self):
        _, train_df, val_df = self.create_dataset()

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df['text'].to_list())

        return tokenizer

    def create_sequences(self, tokenizer, max_length=46):
        df, train_df, val_df = self.create_dataset()

        df_sequences = tokenizer.texts_to_sequences(df['text'].to_list())
        train_sequences = tokenizer.texts_to_sequences(train_df['text'].to_list())
        val_sequences = tokenizer.texts_to_sequences(val_df['text'].to_list())

        df_padded_sequences = pad_sequences(df_sequences, maxlen=max_length, padding='post', truncating='post')
        train_padded_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
        val_padded_sequences = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')

        return df_padded_sequences, train_padded_sequences, val_padded_sequences

    def create_sequences_aspect(self, tokenizer, max_length=8):
        df, train_df, val_df = self.create_dataset()

        df_aspects = tokenizer.texts_to_sequences(df['aspect'].to_list())
        train_aspects = tokenizer.texts_to_sequences(train_df['aspect'].to_list())
        val_aspects = tokenizer.texts_to_sequences(val_df['aspect'].to_list())

        df_padded_aspects = pad_sequences(df_aspects, maxlen=max_length, padding='post', truncating='post')
        train_padded_aspects = pad_sequences(train_aspects, maxlen=max_length, padding='post', truncating='post')
        val_padded_aspects = pad_sequences(val_aspects, maxlen=max_length, padding='post', truncating='post')

        return df_padded_aspects, train_padded_aspects, val_padded_aspects

    def create_embedding_matrix(self, tokenizer, glove_path, embedding_dim):

        vocab_size = len(tokenizer.word_index) + 1

        path_to_glove = glove_path

        embeddings_index = {}
        with open(path_to_glove) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = tf.random.uniform(shape=(300,), minval=-0.25, maxval=0.25).numpy()

        return embedding_matrix

    def create_labels(self):
        df, train_df, val_df = self.create_dataset()
        
        df_labels = df['label'].to_numpy()
        train_labels = train_df['label'].to_numpy()
        val_labels = val_df['label'].to_numpy()

        ohe = OneHotEncoder(sparse=False)
        train_ohe_labels = ohe.fit_transform(train_labels.reshape(-1, 1))
        df_ohe_labels = ohe.transform(df_labels.reshape(-1, 1))
        val_ohe_labels = ohe.transform(val_labels.reshape(-1, 1))

        return df_ohe_labels, train_ohe_labels, val_ohe_labels

    def create_pw(self, max_length=46):
        df, _, _ = self.create_dataset()

        positionW = []
        for i in range(0, len(df)):
            positionW.append(loop_pw(df['text'][i], df['aspect'][i]))

        for i in range(len(positionW)):
            if len(positionW[i]) < max_length:
                pad_length = max_length - len(positionW[i])
                positionW[i] = np.pad(positionW[i], (0, pad_length), 'constant')

            else:
                    positionW[i] = positionW[i][:max_length]

        positionW = np.stack(arrays=positionW, axis=0)

        return positionW