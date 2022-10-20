import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.rnn import RNNCell


class RNN(RNNCell):
    
    def call(self, inp, **kwargs):
        pass

    def __init__(self, hidden_units_gru, nodes_total, input_size=None,
                 activ=tf.nn.tanh, reuse=None):
        
        super(RNN, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._activ = activ
        self._nodes_total = nodes_total
        self._hidden_units_gru = hidden_units_gru



    @property
    def state_size(self):
        return self._nodes_total * self._hidden_units_gru

    @property
    def output_size(self):
        return self._hidden_units_gru

    def __call__(self, inp, state, scope=None):

        with tf.variable_scope(scope or "gru"):
            with tf.variable_scope("gates"): 
                # GRU Gates.
                val = tf.nn.sigmoid(
                    self._linear(inp, state, 2 * self._hidden_units_gru, bias=1.0, scope=scope))
                res, upd = tf.split(val, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                cl = self._linear(inp, res * state, self._hidden_units_gru, scope=scope)
                if self._activ is not None:
                    cl = self._activ(cl)
            h_new = upd * state + (1 - upd) * cl
        return h_new, h_new
        
    
    def _linear(self, inp, state, size_out, bias=0.0, scope=None):
        inp = tf.expand_dims(inp, 2)
       
        stt = tf.reshape(state, (-1, self._nodes_total, self._hidden_units_gru))
        
        x_h  = tf.concat([inp, stt], axis=2)
        size_inp = x_h.get_shape()[2].value
        
        a = tf.reshape(x_h, shape=[-1, size_inp])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            model_weights = tf.get_variable(
                'weights', [size_inp, size_out], initializer=tf.contrib.layers.xavier_initializer())
            model_biases = tf.get_variable(
                "biases", [size_out], initializer=tf.constant_initializer(bias))

            a = tf.matmul(a, model_weights)         
            a = tf.nn.bias_add(a, model_biases)

            a = tf.reshape(a, shape=[-1, self._nodes_total ,size_out])
            a = tf.reshape(a, shape=[-1, self._nodes_total * size_out])
        return a  
