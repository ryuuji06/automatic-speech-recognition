
# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import numpy as np
import pickle

from feature_extraction import audio_spectrogram, audio_mfcc, text_to_int_sequence, int_sequence_to_text
from data_manager import DatasetManager
import rnn_models as mods
#from rnn_models import ctc_lambda_func, add_ctc_loss, simple_rnn_model

from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   

from keras import backend as K
from tensorflow import constant # to convert numpy to tf.Tensor

import matplotlib.pyplot as plt


# =================================================================
# AUTOMATIC SPEECH RECOGNITION WITH CTC-RNN
# TESTING DIFFERENT MODEL ARCHITECTURES
# =================================================================
# exec(open('asr_ctc_train.py').read())

# DESCRIPTION
#  - implement ASR using RNN with the Connectionist Temporal Classification (CTC)
#  - CTC is suitable for problems in which the input sequence has length
# different from the output sequence

# IMPLEMENTATION
#  - CTC computes a different loss, that can not be used in the conventional Keras framework
#  - CTC loss is computed as the model output during training and evaluation, with lambda layer
#  - we must use separate models for prediction and for training/evaluation

# PROBLEMS when loading models with Lambda Layer
# (1) either saving as H5 file
# (2) either as the keras savemodel (folder)
# (even referencing lambda layer to custom_objects)
#  - solution: save only model weights

# TESTS UNTILL NOW
# (1) Two LSTM layers, followed by TimeDistributed Dense (standard test)
# (2) decreasing and increasing number of hidden units
# (3) decrease and increase learning rate
# (4) use batch normalization
#  - after each LSTM layer
#    (becomes very unstable if increase learning rate)
#    (try different value for gradient clipping - reduce clipping norm)
#  - test using only before dense

# (5) test different learning rate decays

# (6) substitute LSTM by GRU

# (7) precede RNN by CNN




feature_type = 'mfcc' # spectrogram or mfcc
model = 2
test = 1
weightsfile = '/weights-17-186.486.h5'
folder = 'results/%s_model%02.d_test%02.d'%(feature_type,model,test)

layer_type, num_layers, n_units = 'LSTM', 2, [128,128]
batch_normalization = False
drop, drop_rates = False, [0.3]


# ---------------------------------------------------
print('\n(1) Load Dataset and set preprocessing')
# ---------------------------------------------------
# load data created during training, recreate data manager

# load simulation parameters
with open(folder+'/params.pickle', 'rb') as handle:
   params = pickle.load(handle)

if params['feature_type'] == 'spectrogram':
	feat_dim = params['feat_dim']
	dataset = DatasetManager(minibatch_size=params['minibatch_size'],
	            test_file=params['test_file'],
	            feat_size=feat_dim,
	            feature_func=lambda path: audio_spectrogram(path, params['max_freq'], params['window_width'], params['window_shift']),
	            label_func=text_to_int_sequence)
elif params['feature_type'] == 'mfcc':
	feat_dim = params['feat_dim']
	dataset = DatasetManager(minibatch_size=params['minibatch_size'],
	            test_file=params['test_file'],
	            feat_size=feat_dim,
	            feature_func=lambda path: audio_mfcc(path, params['feat_dim'], params['window_width'], params['window_shift']),
	            label_func=text_to_int_sequence)
dataset.set_feature_normalization( precomputed=(params['feats_mean'],params['feats_std']) )
print('There are %d total testing examples.' % len(dataset.test_input_paths))

# Optimizer
# in RNN, it is usually necessary to use gradient clipping
optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

# Define Network Model
if model == 1:
	# single GRU layer (performs realy bad)
	# model_loss, model_pred = model_0(input_dim=feat_dim, optimizer=optimizer)
	model_loss, model_pred = mods.CTC_simpleRNN(feat_dim,
		num_layers=1, num_units=[29], rnn_type='LSTM',
    	batch_norm=False, drop=False, drop_rates=[1.0], optimizer=optimizer)
elif model == 2:
	# Double LSTM layers, TimeDistributed Dense
	# model_loss, model_pred = model_1(input_dim=feat_dim, optimizer=optimizer)
	model_loss, model_pred = mods.CTC_RNN_TimeDistrib(feat_dim, 29,
		num_layers=num_layers, num_units=n_units, rnn_type=layer_type,
		batch_norm=batch_normalization, drop=drop, drop_rates=drop_rates, optimizer=optimizer)
elif model == 3:
	# Double LSTM layer, TimeDistributedDense, with Batch Normalization
	# model_loss, model_pred = model_2(input_dim=feat_dim, optimizer=optimizer)
	model_loss, model_pred = mods.CTC_BiRNN_TimeDistrib(feat_dim, 29,
		num_layers=num_layers, num_units=n_units, rnn_type=layer_type,
		batch_norm=batch_normalization, drop=drop, drop_rates=drop_rates, optimizer=optimizer)
elif model == 4:
	# Double Bi-LSTM layer, TimeDistributedDense, with Batch Normalization
	# model_loss, model_pred = model_3(input_dim=feat_dim, optimizer=optimizer)
	model_loss, model_pred = mods.CTC_CNN_BiRNN_TimeDistrib(feat_dim, 29,
    	num_layers_cnn=2, num_cnn_filters=[32,32], conv_kernel=5, conv_stride=3,
    	num_layers_rnn=2, num_units=[128,128], rnn_type='LSTM',
    	batch_norm=False, drop=False, drop_rates=[1.0], optimizer=optimizer)


model_loss.load_weights(folder+weightsfile)
model_pred.load_weights(folder+weightsfile)


# ---------------------------------------------------
print('\n(2) Prediction and Decoding')
# ---------------------------------------------------

# SELECT SAMPLE
# number of the sample to be taken from dataset.test_input_paths
sample = 5

samp_input = dataset.featurize(dataset.test_input_paths[sample])
samp_label = dataset.labelize(dataset.test_labels[sample])
sh1 = samp_input.shape
sh2 = samp_label.shape
samp_input = samp_input.reshape(1,sh1[0],sh1[1])
samp_label = samp_label.reshape(1,sh2[0])
input_len = np.array([samp_input.shape[1]]).reshape(1)
label_len = np.array([samp_label.shape[1]]).reshape(1)
print('\nSelected sample:',sample)
print('Input feature shape:',samp_input.shape)
print('Target text:',''.join(int_sequence_to_text(samp_label[0]+1)))
print('Target length:',samp_label.shape)


# NETWORK OUTPUT (LOSS AND TOKEN PROBABILITIES)

print('\nModel loss: ',model_loss.predict([ samp_input, samp_label, input_len, label_len ]))
#pred_probs = model_pred.predict(samp_input) # this returns a numpy array
pred_probs = model_pred(samp_input) # this returns a tf.Tensor



# # (1) DECODING BY HAND
# # (convert probabilities in deterministic sequence, and remove null element)
# pred_seq_1 = [np.argmax(pred_probs.numpy()[0][i]) for i in range(input_len[0])]
# # remove null charcters
# cont = True
# while cont:
# 	try:
# 		pred_seq_1.remove(28)
# 	except ValueError:
# 		cont = False
# print('Prediction by hand')
# print( ''.join(int_sequence_to_text(np.array(pred_seq_1)+1)) )


# (2) USING KERAS DECODING

# greedy search
pred_seq_2 = K.ctc_decode(pred_probs, constant(input_len))
print('Greedy prediction')
p = pred_seq_2[0][0][0].numpy()
print( ''.join(int_sequence_to_text( p[p!=-1]+1 )) )
# beam search
pred_seq_3 = K.ctc_decode(pred_probs, constant(input_len), greedy=False, top_paths=5 )
print('Beam prediction')
for i in range(5):
	x = pred_seq_3[0][i][0].numpy()
	print(''.join(int_sequence_to_text( x[x!=-1]+1 )) )


# plt.figure(1)
# plt.subplot(121)
# plt.plot(a2[0],'o-')
# plt.show()



# ---------------------------------------------------------------