import tensorflow as tf
from tensorflow.keras.layers import Layer


class CPT(Layer):
    def __init__(self, hidden_nums):
        super(CPT, self).__init__()
        self.hidden_nums = hidden_nums

        self.t_weights = {
            'trans_weights': tf.Variable(tf.initializers.RandomUniform(-0.01, 0.01)(shape=[4 * self.hidden_nums, 2*self.hidden_nums]),
                                         trainable=True,
                                         import_scope='TST_weights',
                                         name='trans_W')
        }
        self.t_bias = {
            'trans_bias': tf.Variable(tf.zeros_initializer()(shape=[2 * self.hidden_nums]),
                                      trainable=True,
                                      import_scope='TST_bias',
                                      name='trans_b')
        }

    def tst(self, target_hidden_states, hidden_states):
        hidden_sp = tf.shape(hidden_states)
        batch_size = hidden_sp[0]

        # (seq_len , batch_size, 2 * hidden_size)
        hs_ = tf.transpose(hidden_states, perm=[1, 0, 2])
        # (batch_size, 2*hidden_size, target_len)
        t_ = tf.transpose(target_hidden_states, perm=[0, 2, 1])

        # tst
        sentence_index = 0
        sentence_array = tf.TensorArray(
            dtype=tf.float32, size=1, dynamic_size=True)

        def body(sentence_index, sentence_array):
            # (batch_size, 2*hidden_size)
            hi = tf.transpose(tf.gather_nd(
                hs_, [[sentence_index]]), perm=[1, 2, 0])

            # (batch_size, target_length)
            ai = tf.nn.softmax(tf.squeeze(
                tf.matmul(target_hidden_states, hi), axis=-1))

            # (batch_size, 2 * hidden_size, 1)
            ti = tf.matmul(t_, tf.expand_dims(ai, axis=-1))

            hi = tf.squeeze(hi, axis=-1)
            ti = tf.squeeze(ti, axis=-1)

            # concatenate (batch_size, 1, 4 * hidden_size)
            concated_hi = tf.concat([hi, ti], axis=-1)
            concated_hi = tf.reshape(
                concated_hi, [batch_size, 1, 4 * self.hidden_nums])

            hi_new = tf.math.tanh(
                tf.matmul(concated_hi, tf.tile(tf.expand_dims(self.t_weights['trans_weights'], axis=0),
                                               [batch_size, 1, 1])) + self.t_bias['trans_bias']
            )

            hi_new = tf.squeeze(hi_new, axis=1)

            sentence_array = sentence_array.write(sentence_index, hi_new)

            return (sentence_index + 1, sentence_array)

        def cond(sentence_index, sentence_array):
            return sentence_index < hidden_sp[1]

        _, sentence_array = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[sentence_index, sentence_array])


        sentence_array = tf.transpose(sentence_array.stack(), perm=[1, 0, 2])

        return sentence_array

    def lf_layer(self, target_hidden_states, hidden_states):
        hidden_states_ = self.tst(target_hidden_states, hidden_states)

        return hidden_states_ + hidden_states

    def call(self, target_hidden_states, hidden_states):
        """
        Input : {
            target_embeddings: (?, ?, embed_dim),
            target_sequence_length : (?, ),
            hidden_states: (?, ?, 2 * hidden_nums)
        }
        """

        output = self.lf_layer(target_hidden_states, hidden_states)

        return output

    def get_config(self):
        config = super(CPT, self).get_config()
        config.update({
            'hidden_nums': self.hidden_nums,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
