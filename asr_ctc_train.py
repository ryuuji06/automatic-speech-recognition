
# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import numpy as np
import pickle
from time import time

from feature_extraction import audio_spectrogram, audio_mfcc, fft_size, text_to_int_sequence
from data_manager import DatasetManager
import rnn_models as mods
#from rnn_models import ctc_lambda_func, add_ctc_loss, simple_rnn_model

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint   

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



# -----------------------------------------------------


feature_type = 'mfcc' # spectrogram or mfcc
model = 2
epochs = 30

layer_type, num_layers, n_units = 'LSTM', 2, [128,128]
minibatch_size = 32
batch_normalization = False
drop, drop_rates = False, [0.3]

# Optimizer
optimizer=SGD(lr=0.02, decay=1e-1, momentum=0.9, nesterov=True, clipnorm=5)
#optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

test = 10
descrip = f'Testing recurrent layer with TimeDistributed Dense layer.\n\
{layer_type} layers, {num_layers} layers with {n_units} units in recurrent hidden layers.\n\
Batch size {minibatch_size}, batch normalization {batch_normalization}, dropout {drop}. SGD optimizer.\n\
\nTime inverse decay in learning rate of 1e-1.'

# make directories, if necessary
folder = 'results/%s_model%02.d_test%02.d'%(feature_type,model,test)
if not os.path.exists('results'):
    os.makedirs('results')
if os.path.exists(folder):
	raise AssertionError('Folder already exists. Be sure to use a proper name for the test.')
else:
    os.makedirs(folder)

with open(folder+'/description.txt', 'w') as file:
	file.write(descrip)

# Preprocessing parameters
window_shift = 0.010 # in sec
window_width = 0.020 # in sec
max_freq = 8000 # max freq shown in spectrogram
sample_rate = 16000
# feature length
mfcc_dim = 13 # number of mfcc parameters
fft_dim = fft_size(max_freq,sample_rate,window_width)

if window_shift > window_width:
    raise ValueError("step size must not be greater than window size")


# ---------------------------------------------------
print('\n(1) Load Dataset and set preprocessing')
# ---------------------------------------------------

# path to JSON file
train_file = 'corpus_clean_dev.json'
test_file = 'corpus_clean_test.json'

if feature_type == 'spectrogram':
	feat_dim = fft_dim
	dataset = DatasetManager(minibatch_size=20,
	            train_file=train_file,
	            feat_size=fft_dim,
	            feature_func=lambda path: audio_spectrogram(path, max_freq, window_width, window_shift),
	            label_func=text_to_int_sequence)
elif feature_type == 'mfcc':
	feat_dim = mfcc_dim
	dataset = DatasetManager(minibatch_size=minibatch_size,
	            train_file=train_file,
	            feat_size=mfcc_dim,
	            feature_func=lambda path: audio_mfcc(path, mfcc_dim, window_width, window_shift),
	            label_func=text_to_int_sequence)

dataset.load_data(test_file,'validation')
dataset.set_feature_normalization()

params = {'feature_type': feature_type,
		   'window_shift':window_shift,
		   'window_width':window_width,
		   'max_freq':max_freq,
		   'feat_dim':feat_dim,
		   'minibatch_size':minibatch_size,
		   'train_file':train_file,
		   'test_file':test_file,
		   'feats_mean':dataset.feats_mean,
		   'feats_std':dataset.feats_std}

# length of training and validation sets
print('There are %d total training examples.' % len(dataset.train_input_paths))
print('There are %d total validating examples.' % len(dataset.valid_input_paths))

with open(folder+'/params.pickle', 'wb') as handle:
	pickle.dump(params, handle)

#with open('results/%s_model_%d_params.pickle'%(feature_type,model_test), 'rb') as handle:
#    dt2 = pickle.load(handle)



# ---------------------------------------------------
print('\n(2) Define model')
# ---------------------------------------------------



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

model_loss.summary()

#TEST PREDICTION FOR BOTH MODELS
sample = 1
samp_input = dataset.featurize(dataset.train_input_paths[sample])
samp_label = dataset.labelize(dataset.train_labels[sample])
sh1 = samp_input.shape
sh2 = samp_label.shape
samp_input = samp_input.reshape(1,sh1[0],sh1[1])
samp_label = samp_label.reshape(1,sh2[0])
input_len = np.array([samp_input.shape[1]]).reshape(1,1)
label_len = np.array([samp_label.shape[1]]).reshape(1,1)
print(samp_input.shape)
print(samp_label.shape)

a1 = model_loss.predict([ samp_input, samp_label, input_len, label_len ])
a2 = model_pred.predict([samp_input])



# ---------------------------------------------------
print('\n(3) Train Model')
# ---------------------------------------------------

pickle_path = folder+'/hist.pickle'
save_model1_path = folder+'/weights-{epoch:02d}-{val_loss:.3f}.h5'
#save_model2_path='results/%s_model_%d_pred.h5'%(feature_type,model_test)

# calculate steps_per_epoch
num_train_samples=len(dataset.train_input_paths)
train_steps = num_train_samples//minibatch_size
# calculate validation_steps
num_valid_samples = len(dataset.valid_input_paths) 
valid_steps = num_valid_samples//minibatch_size



# add checkpointer
checkpointer = ModelCheckpoint(save_model1_path, save_best_only=True, save_weights_only=True, verbose=0)

# train the model
t1 = time()
hist = model_loss.fit_generator(generator=dataset.next_train(), steps_per_epoch=train_steps,
    epochs=epochs, validation_data=dataset.next_valid(), validation_steps=valid_steps,
    callbacks=[checkpointer], verbose=1)

# save model loss history
with open(pickle_path, 'wb') as f:
    pickle.dump(hist.history, f)


t2 = time()
print('Elapsed time: %.3fs'%(t2-t1)); t1 = t2


#TEST PREDICTION FOR BOTH MODELS
sample = 1
samp_input = dataset.featurize(dataset.train_input_paths[sample])
samp_label = dataset.labelize(dataset.train_labels[sample])
sh1 = samp_input.shape
sh2 = samp_label.shape
samp_input = samp_input.reshape(1,sh1[0],sh1[1])
samp_label = samp_label.reshape(1,sh2[0])
input_len = np.array([samp_input.shape[1]]).reshape(1,1)
label_len = np.array([samp_label.shape[1]]).reshape(1,1)
print(samp_input.shape)
print(samp_label.shape)

b1 = model_loss.predict([ samp_input, samp_label, input_len, label_len ])
b2 = model_pred.predict([samp_input])

plt.figure(1)
plt.subplot(121)
plt.plot(a2[0],'o-')
plt.subplot(122)
plt.plot(b2[0],'o-')
plt.show()




# ---------------------------------------------------------------