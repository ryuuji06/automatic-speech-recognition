import numpy as np
from python_speech_features import mfcc
from numpy.lib.stride_tricks import as_strided
import scipy.io.wavfile as wav
#from tensorflow.keras.utils import to_categorical



# ------------------------------------------------
# COMPUTING FEATURES FROM RAW DATA ARRAY
# ------------------------------------------------

def my_spectrogram(signal, sample_rate=16000, max_freq=8000, step=0.010, window=0.020, scale_mode=1, eps=1e-14):
    #max_freq = sample_rate / 2
    # compute step and window lengths in samples
    hop_length = int( step * sample_rate)
    fft_length = int( window * sample_rate)

    window = np.hanning(fft_length)[:, None]

    # truncate signal so that there is no remainder when windows sliding
    trunc = (len(signal) - fft_length) % hop_length
    x = signal[:len(signal) - trunc]

    # reshape x, dealing with byte reallocation
    # as_strided itself only makes reference to the source array. But here we are overwriting it
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length) # deals with byte reallocation
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    #assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns, and take magnitude
    x = np.fft.rfft(x * window, axis=0) # FFT for real input
    x = np.absolute(x)

    # The scaling below follows the convention of matplotlib.mlab.specgram
    # 2.0 for everything except dc and fft_length/2
    if scale_mode==1:
        x = x**2
        window_norm = np.sum(window**2)
        scale = window_norm * sample_rate
        x[1:-1, :] *= (2.0 / scale)
        x[(0, -1), :] /= scale
    
    # take only the selected frequencies, and take log
    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose( np.log( x[:ind, :] + eps ) )


def fft_size(max_freq, sample_rate, window):
    return np.int32(np.floor( max_freq * int( window * sample_rate) / sample_rate )) + 1



# ------------------------------------------------
# READING RAW DATA FROM FILES
# ------------------------------------------------
# return feature array from file path 

def audio_spectrogram(filepath, max_freq=8000, window_width=0.02, window_shift=0.01):
    """ For a given file path (audio file), calculate the corresponding feature (spectrogram)"""
    sampling_rate, raw_audio = wav.read(filepath)
    return my_spectrogram(raw_audio, sample_rate=sampling_rate, max_freq=max_freq,
                        window=window_width, step=window_shift )

def audio_mfcc(filepath, mfcc_dim=13, window_width=0.02, window_shift=0.01):
    sampling_rate, raw_audio = wav.read(filepath)
    return mfcc(raw_audio, samplerate=sampling_rate,
                    winlen=window_width, winstep=window_shift, numcep=mfcc_dim, preemph=0.97)



# ------------------------------------------------
# PROCESSING LABELS (convert to numerical values)
# ------------------------------------------------

# letter are encoded into 29 integers
# 26 actual letters, apostrophe, space and null element
# (the "blank" character is mapped to 28)

char_map_str = """
' 0
<SPACE> 1
a 2
b 3
c 4
d 5
e 6
f 7
g 8
h 9
i 10
j 11
k 12
l 13
m 14
n 15
o 16
p 17
q 18
r 19
s 20
t 21
u 22
v 23
w 24
x 25
y 26
z 27
"""

# strip(): remove some heading and trailing entries of the string
# split(): split string into list; we can specify separator

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)+1] = ch
index_map[2] = ' '


def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return np.array(int_sequence)
    #return to_categorical(int_sequence, num_classes=29)

def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = index_map[c]
        text.append(ch)
    return text