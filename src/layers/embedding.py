import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Embedding


def embed_layer(num_tokens, embedding_dim, embedding_matrix):

    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        mask_zero=True,
        trainable=False,
    )

    return embedding_layer


def BiLSTM(hidden_dims):

    lstm_layer = Bidirectional(LSTM(hidden_dims, return_sequences=True))

    return lstm_layer


def position_embedding(hs, pw):
        """
        @hs: (batch_size, sentence_length, 2 * hidden_nums)
        @pw: (batch_size, sentence_length)
        """
        weighted_hs = hs * tf.expand_dims(pw, axis=-1)

        return weighted_hs
