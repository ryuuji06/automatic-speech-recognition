# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import numpy as np
import pickle
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

# -------------------------------------------------



feature_type = 'mfcc' # spectrogram or mfcc
model = 2
tests = ['01','01_variant'] # variation of the same experiment
#tests = ['02','03','01'] # varying units in LSTM
#tests = ['05','01','04'] # varying learning rate
#tests = ['08','01','09','10'] # varying learning rate decay
#tests = ['01','06'] # batch normalized and non batch normalized
#tests = ['04','07','07_try1'] # batch normalized

# ---------------------------------------------------
print('\n(1) Load history')
# ---------------------------------------------------

K = len(tests)
hist = []

for k in range(K):

	folder = 'results/%s_model%02.d_test%s'%(feature_type,model,tests[k])
	# load history data
	with open(folder+'/hist.pickle', 'rb') as handle:
	    hist.append(pickle.load(handle))


# ---------------------------------------------------
print('\n(2) Plot performance curves')
# ---------------------------------------------------

plt.figure(1)

for k in range(K):

	plt.subplot(121); plt.grid(); plt.title('Loss')
	plt.plot(hist[k]['loss'],'o-')

	plt.subplot(122); plt.grid(); plt.title('Validation loss')
	plt.plot(hist[k]['val_loss'],'o-')

plt.show()

plt.figure(2); plt.grid()
plt.plot(10*np.log10(hist[0]['loss']))
plt.plot(10*np.log10(hist[0]['val_loss']),'o')
plt.legend(['Training loss','Validation loss'])
plt.ylabel('Loss (dB)')
plt.xlabel('Epochs')



# ---------------------------------------------------------------