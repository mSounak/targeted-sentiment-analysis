import tensorflow as tf
from layers.embedding import embed_layer, BiLSTM, position_embedding
from layers.cpt import CPT
from layers.output import CnnLayer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout

def tnet_lf(num_tokens, embedding_matrix, embedding_dim=300, hidden_dims=50, filter_nums=50, dropout_rate=0.3, num_classes=3):

    # sentence input
    sentence_input = Input(shape=(None, ), name='sentence_input')
    sentence_embed = embed_layer(num_tokens, embedding_dim, embedding_matrix)(sentence_input)
    sentence_hidden_states = BiLSTM(hidden_dims)(sentence_embed)
    sentence_hidden_states = tf.keras.layers.Dropout(dropout_rate)(sentence_hidden_states)

    # target input
    target_input = Input(shape=(None, ), name='target_input')
    target_embed = embed_layer(num_tokens, embedding_dim, embedding_matrix)(target_input)
    target_hidden_states = BiLSTM(hidden_dims)(target_embed)
    target_hidden_states = tf.keras.layers.Dropout(dropout_rate)(target_hidden_states)

    # CPT 1
    cpt_out_1 = CPT(hidden_dims)(target_hidden_states, sentence_hidden_states)

    # position weighting
    pw = Input(shape=(None,), name='pw')
    modified_hidden_states_1 = position_embedding(cpt_out_1, pw)

    # CPT 2
    cpt_out_2 = CPT(hidden_dims)(target_hidden_states, modified_hidden_states_1)

    # position weighting
    modified_hidden_states_2 = position_embedding(cpt_out_2, pw)

    # CNN layer
    cnn_output, cnn_features = CnnLayer(filter_nums, 3)(modified_hidden_states_2)

    # Dropout
    drp_out = tf.keras.layers.Dropout(dropout_rate)(cnn_output)

    # output layer
    output_layer = Dense(num_classes, activation='softmax')(drp_out)

    model = Model(inputs=[sentence_input, target_input, pw], outputs=output_layer)

    return model