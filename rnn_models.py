
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Lambda, Activation, Dense
from keras.layers import SimpleRNN, GRU, LSTM, TimeDistributed, BatchNormalization, Bidirectional, Dropout
from keras.backend import ctc_batch_cost # for CTC
#from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
#    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)


# ------------------------------------
# AUXILIAR FOR CTC LOSS
# ------------------------------------

# auxiliar function to wrap ctc_batch_cost() to a lambda layer
# (labels must be received as integers, not one-hot encoded vectors)
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


# ------------------------------------
# ORIGINAL IMPLEMENTATIONS
# ------------------------------------
# implement CTC separately from the model

def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    return model


def simple_rnn_model(input_dim, output_dim=29):
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    
    return model



# =============================================
#   C T C - R N N   M O D E L S
# =============================================
# these functions return two models:
# one for computing loss, other for prediction (network output)


def model_0(input_dim, output_dim=29, optimizer=Adam()):
    """
    MODEL 0: SINGLE RECURSIVE LAYER (works really bad)
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    simp_rnn = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )

    return model_loss, model_pred



def model_1(input_dim, output_dim=29, optimizer=Adam()):
    """
    MODEL 1: TWO RNN LAYERS AND TIME DISTRIBUTED LAYER
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    lstm0 = LSTM(128, return_sequences=True)(input_data)
    lstm1 = LSTM(128, return_sequences=True)(lstm0)
    dense = TimeDistributed(Dense(output_dim))(lstm1)
    y_pred = Activation('softmax', name='softmax')(dense)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )
    
    return model_loss, model_pred



def model_2(input_dim, output_dim=29, optimizer=Adam()):
    """
    MODEL 2: TWO RNN LAYERS, TIME DISTRIBUTED LAYER AND BATCH NORMALIZATION
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    lstm0 = LSTM(128, return_sequences=True)(input_data)
    bn1 = BatchNormalization(axis=-1)(lstm0)
    lstm1 = LSTM(128, return_sequences=True)(bn1)
    bn2 = BatchNormalization(axis=-1)(lstm1)
    dense = TimeDistributed(Dense(output_dim))(bn2)
    y_pred = Activation('softmax', name='softmax')(dense)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )
    
    return model_loss, model_pred



def model_3(input_dim, output_dim=29, optimizer=Adam()):
    """
    MODEL 3: TWO BIDIRECTIONAL RNN LAYERS, TIME DISTRIBUTED LAYER AND BATCH NORMALIZATION
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    lstm0 = Bidirectional(LSTM(128, return_sequences=True))(input_data)
    bn1 = BatchNormalization(axis=-1)(lstm0)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(bn1)
    bn2 = BatchNormalization(axis=-1)(lstm1)
    dense = TimeDistributed(Dense(output_dim))(bn2)
    y_pred = Activation('softmax', name='softmax')(dense)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )
    
    return model_loss, model_pred



# =============================================
#   F L E X I B L E   M O D E L S
# =============================================

# Sole GRU/LSTM layers (batchnorm or not)
# GRU/LSTM layers followed by TimeDistributed Dense (batchnorm or not)
# (test the difference of using Batch normalization, with different learning rate)

# Bidir GRU/LSTM layers followed by TimeDistributed (batchnorm or not)

# Conv1D, Bidir GRU/LSTM layers followed by TimeDistributed (batchnorm or not)
# Conv1D, Bidir GRU/LSTM layers with dropout, followed by TimeDistributed (batchnorm or not)


def CTC_simpleRNN(input_dim, num_layers=1, num_units=[128], rnn_type='LSTM',
    batch_norm=False, drop=False, drop_rates=[1.0], optimizer=Adam()):
    """
    Only RNN (GRU or LSTM) layers
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    x = input_data
    for i in range(num_layers):
        if rnn_type  == 'LSTM':
            x = LSTM(num_units[i], return_sequences=True)(x)
        elif rnn_type  == 'GRU':
            x = GRU(num_units[i], return_sequences=True)(x)
        if drop:
            x = Dropout(rate=drop_rates[i])(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
    y_pred = Activation('softmax', name='softmax')(x)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )

    return model_loss, model_pred


def CTC_RNN_TimeDistrib(input_dim, output_dim, num_layers=1, num_units=[128],
    rnn_type='LSTM', batch_norm=False, drop=False, drop_rates=[1.0], optimizer=Adam()):
    """
    Only RNN (GRU or LSTM) layers, with final time distributed dense layer
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    x = input_data
    for i in range(num_layers):
        if rnn_type  == 'LSTM':
            x = LSTM(num_units[i], return_sequences=True)(x)
        elif rnn_type  == 'GRU':
            x = GRU(num_units[i], return_sequences=True)(x)
        if drop:
            x = Dropout(rate=drop_rates[i])(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
    dense = TimeDistributed(Dense(output_dim))(x)
    y_pred = Activation('softmax', name='softmax')(dense)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )

    return model_loss, model_pred


def CTC_BiRNN_TimeDistrib(input_dim, output_dim, num_layers=1, num_units=[128],
    rnn_type='LSTM', batch_norm=False, drop=False, drop_rates=[1.0], optimizer=Adam()):
    """
    Only RNN (GRU or LSTM) layers, with final time distributed dense layer
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    x = input_data
    for i in range(num_layers):
        if rnn_type  == 'LSTM':
            x = Bidirectional(LSTM(num_units[i], return_sequences=True))(x)
        elif rnn_type  == 'GRU':
            x = Bidirectional(GRU(num_units[i], return_sequences=True))(x)
        if drop:
            x = Dropout(rate=drop_rates[i])(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
    dense = TimeDistributed(Dense(output_dim))(x)
    y_pred = Activation('softmax', name='softmax')(dense)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )

    return model_loss, model_pred


def CTC_CNN_BiRNN_TimeDistrib(input_dim, output_dim,
    num_layers_cnn=1, num_cnn_filters=[32], conv_kernel=5, conv_stride=3,
    num_layers_rnn=1, num_units=[128], rnn_type='LSTM',
    batch_norm=False, drop=False, drop_rates=[1.0], optimizer=Adam()):
    """
    Only RNN (GRU or LSTM) layers, with final time distributed dense layer
    """

    # Process acoustic input (RNN model)
    input_data = Input(name='input_seq', shape=(None, input_dim)) # acoustic input data
    x = input_data
    for i in range(num_layers_cnn):
        x = Conv1D(num_cnn_filters[i], conv_kernel, strides=conv_stride, activation='relu')(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
    for i in range(num_layers):
        if rnn_type  == 'LSTM':
            x = Bidirectional(LSTM(num_units[i], return_sequences=True))(x)
        elif rnn_type  == 'GRU':
            x = Bidirectional(GRU(num_units[i], return_sequences=True))(x)
        if drop:
            x = Dropout(rate=drop_rates[i])(x)
        if batch_norm:
            x = BatchNormalization(axis=-1)(x)
    dense = TimeDistributed(Dense(output_dim))(x)
    y_pred = Activation('softmax', name='softmax')(dense)

    # CTC loss is implemented in a lambda layer
    true_labels = Input(name='label_seq', shape=(None,), dtype='float32') # linguistic label
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, true_labels, input_lengths, label_lengths])

    # define models for loss (train and evaluate) and for prediction
    model_loss = Model(
        inputs=[input_data, true_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    model_pred = Model( inputs=input_data, outputs=y_pred )
    
    # insert optimizer
    model_loss.compile( loss={'ctc': lambda y_t, y_p: y_p}, optimizer=optimizer )
    model_pred.compile( optimizer=optimizer )

    return model_loss, model_pred


# ------------------------------------
#  O T H E R S
# ------------------------------------

def conv_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

# def rnn_model(input_dim, units, activation, output_dim=29):
#     """ Build a recurrent network for speech 
#     """
#     # Main acoustic input
#     input_data = Input(name='the_input', shape=(None, input_dim))
#     # Add recurrent layer
#     simp_rnn = GRU(units, activation=activation,
#         return_sequences=True, implementation=2, name='rnn')(input_data)
#     # TODO: Add batch normalization 
#     bn_rnn = ...
#     # TODO: Add a TimeDistributed(Dense(output_dim)) layer
#     time_dense = ...
#     # Add softmax activation layer
#     y_pred = Activation('softmax', name='softmax')(time_dense)
#     # Specify the model
#     model = Model(inputs=input_data, outputs=y_pred)
#     model.output_length = lambda x: x
#     print(model.summary())
#     return model


# def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
#     conv_border_mode, units, output_dim=29):
#     """ Build a recurrent + convolutional network for speech 
#     """
#     # Main acoustic input
#     input_data = Input(name='the_input', shape=(None, input_dim))
#     # Add convolutional layer
#     conv_1d = Conv1D(filters, kernel_size, 
#                      strides=conv_stride, 
#                      padding=conv_border_mode,
#                      activation='relu',
#                      name='conv1d')(input_data)
#     # Add batch normalization
#     bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
#     # Add a recurrent layer
#     simp_rnn = SimpleRNN(units, activation='relu',
#         return_sequences=True, implementation=2, name='rnn')(bn_cnn)
#     # TODO: Add batch normalization
#     bn_rnn = ...
#     # TODO: Add a TimeDistributed(Dense(output_dim)) layer
#     time_dense = ...
#     # Add softmax activation layer
#     y_pred = Activation('softmax', name='softmax')(time_dense)
#     # Specify the model
#     model = Model(inputs=input_data, outputs=y_pred)
#     model.output_length = lambda x: cnn_output_length(
#         x, kernel_size, conv_border_mode, conv_stride)
#     print(model.summary())
#     return model

# def cnn_output_length(input_length, filter_size, border_mode, stride,
#                        dilation=1):
#     """ Compute the length of the output sequence after 1D convolution along
#         time. Note that this function is in line with the function used in
#         Convolution1D class from Keras.
#     Params:
#         input_length (int): Length of the input sequence.
#         filter_size (int): Width of the convolution kernel.
#         border_mode (str): Only support `same` or `valid`.
#         stride (int): Stride size used in 1D convolution.
#         dilation (int)
#     """
#     if input_length is None:
#         return None
#     assert border_mode in {'same', 'valid'}
#     dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
#     if border_mode == 'same':
#         output_length = input_length
#     elif border_mode == 'valid':
#         output_length = input_length - dilated_filter_size + 1
#     return (output_length + stride - 1) // stride

# def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
#     """ Build a deep recurrent network for speech 
#     """
#     # Main acoustic input
#     input_data = Input(name='the_input', shape=(None, input_dim))
#     # TODO: Add recurrent layers, each with batch normalization
#     ...
#     # TODO: Add a TimeDistributed(Dense(output_dim)) layer
#     time_dense = ...
#     # Add softmax activation layer
#     y_pred = Activation('softmax', name='softmax')(time_dense)
#     # Specify the model
#     model = Model(inputs=input_data, outputs=y_pred)
#     model.output_length = lambda x: x
#     print(model.summary())
#     return model

# def bidirectional_rnn_model(input_dim, units, output_dim=29):
#     """ Build a bidirectional recurrent network for speech
#     """
#     # Main acoustic input
#     input_data = Input(name='the_input', shape=(None, input_dim))
#     # TODO: Add bidirectional recurrent layer
#     bidir_rnn = ...
#     # TODO: Add a TimeDistributed(Dense(output_dim)) layer
#     time_dense = ...
#     # Add softmax activation layer
#     y_pred = Activation('softmax', name='softmax')(time_dense)
#     # Specify the model
#     model = Model(inputs=input_data, outputs=y_pred)
#     model.output_length = lambda x: x
#     print(model.summary())
#     return model

# def final_model():
#     """ Build a deep network for speech 
#     """
#     # Main acoustic input
#     input_data = Input(name='the_input', shape=(None, input_dim))
#     # TODO: Specify the layers in your network
#     ...
#     # TODO: Add softmax activation layer
#     y_pred = ...
#     # Specify the model
#     model = Model(inputs=input_data, outputs=y_pred)
#     # TODO: Specify model.output_length
#     model.output_length = ...
#     print(model.summary())
#     return model


# # FROM DEEP ASR

# # the input
#     input_data = Input(name='the_input', shape=(None, input_dim), dtype='float32')

#     # Batch normalize
#     bn1 = BatchNormalization(axis=-1, name='BN_1')(input_data)

#     # 1D Convs
#     conv1 = Conv1D(512, 5, strides=1, activation='relu', name='Conv1D_1')(bn1)
#     cbn1 = BatchNormalization(axis=-1, name='CBN_1')(conv1)
#     conv2 = Conv1D(512, 5, strides=1, activation='relu', name='Conv1D_2')(cbn1)
#     cbn2 = BatchNormalization(axis=-1, name='CBN_2')(conv2)
#     conv3 = Conv1D(512, 5, strides=1, activation='relu', name='Conv1D_3')(cbn2)

#     # Batch normalize
#     x = BatchNormalization(axis=-1, name='BN_2')(conv3)

#     # BiRNNs
#     # birnn1 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_1'), merge_mode='sum')(bn2)
#     # birnn2 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_2'), merge_mode='sum')(birnn1)
#     # birnn3 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_3'), merge_mode='sum')(birnn2)
#     # birnn4 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_4'), merge_mode='sum')(birnn3)
#     # birnn5 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_5'), merge_mode='sum')(birnn4)
#     # birnn6 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_6'), merge_mode='sum')(birnn5)
#     # birnn7 = Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_7'), merge_mode='sum')(birnn6)

#     # BiRNNs
#     for i in [1, 2, 3, 4, 5]:
#         recurrent = GRU(units=800,
#                         activation='tanh',
#                         recurrent_activation='sigmoid',
#                         use_bias=True,
#                         return_sequences=True,
#                         reset_after=True,
#                         name=f'gru_{i}')
#         x = Bidirectional(recurrent,
#                           name=f'bidirectional_{i}',
#                           merge_mode='concat')(x)
#         x = Dropout(rate=0.5)(x) if i < 5 else x  # Only between

#     # Batch normalize
#     bn3 = BatchNormalization(axis=-1, name='BN_3')(x)

#     dense = TimeDistributed(Dense(1024, activation='relu', name='FC1'))(bn3)
#     y_pred = TimeDistributed(Dense(output_dim, activation='softmax', name='y_pred'), name='the_output')(dense)



# ------------------------------------------------------