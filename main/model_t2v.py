from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from tcn import TCN
from utils.deepar.loss import quantile_mae


def attention(hidden_states, param):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 256)
    @author: felixhao28.
    :param **kwargs:
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(param['unit'], use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


def T2V_NN(param, q):
    K.clear_session()
    input = Input(shape=(param['lookback'], 1))

    # (batch_size, lookback, 1)->(batch_size, hidden_size)
    tcn = TCN(nb_filters=param['unit'], return_sequences=False)(input)

    # BiLSTM+Attention
    # (batch_size, lookback, t2v_dim+1)->(batch_size, lookback, hidden_size)->(batch_size, hidden_size)
    rnn = Bidirectional(LSTM(param['unit'], return_sequences=True))(input)
    rnn = attention(rnn, param)
    # (batch_size, hidden_size) (batch_size, hidden_size)
    concat = concatenate([tcn, rnn])
    nolinear_output = Dense(1)(concat)

    ar = Lambda(lambda k: k[:, -param['hw']:, :])(input)
    ar = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(ar)
    ar = Lambda(lambda k: K.reshape(k, (-1, param['hw'])))(ar)
    linear_output = Dense(1)(ar)

    output = add([nolinear_output, linear_output])

    m = Model(input, output)
    m.compile(loss=quantile_mae(q), optimizer='adam')

    return m
